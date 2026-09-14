// H3Launcher.cs
//
// Small WinForms status-window launcher for the H3 v2 application.
//
// IMPORTANT: This wrapper deliberately delegates ALL launch logic to
// app\RUN_H3_APP.ps1. That script is the single source of truth for
// validating required files, reading config.json, resolving the ComfyUI
// dir/main.py/python, checking that app_port is free, warning about
// missing model files, setting PYTHONUTF8/PYTHONIOENCODING, opening the
// browser once the server is up, and finally running
// "<python> -X utf8 app\server.py".
//
// This file must NOT reimplement, duplicate, or shortcut any of that
// behavior. Its job is to:
//   1. Find app\RUN_H3_APP.ps1 relative to this executable's own location
//      (not the current working directory).
//   2. Launch powershell.exe against that script with no console window,
//      capturing its stdout/stderr and showing it live in a read-only
//      text box so the user can still see the script's own diagnostics
//      (e.g. "Port 8790 is already in use").
//   3. Put the powershell child (and therefore every descendant it spawns:
//      the app server and, transitively, any ComfyUI it launches) into a
//      Windows Job Object with JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE, so
//      closing this window's job handle tears down the whole tree and
//      frees the ports. Windows Job Objects propagate to processes
//      started later by a job member, so the app server and its ComfyUI
//      child both end up in the same job automatically.
//   4. Never kill anything by scanning ports. A ComfyUI the user started
//      themselves must never be touched -- only processes this launcher
//      itself spawned (and their descendants) are in the job.
//
// This file is saved as UTF-8 WITH BOM so csc.exe correctly reads the
// Japanese string literals used for the UI text below.

using System;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Net.Sockets;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.RegularExpressions;
using System.Windows.Forms;

namespace H3Launcher
{
    internal static class Program
    {
        [STAThread]
        private static void Main()
        {
            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);
            Application.Run(new MainForm());
        }
    }

    /// <summary>
    /// Thin wrapper around a Windows Job Object configured to kill every
    /// process in the job as soon as the job handle is closed.
    /// </summary>
    internal sealed class JobObject : IDisposable
    {
        [StructLayout(LayoutKind.Sequential)]
        private struct JOBOBJECT_BASIC_LIMIT_INFORMATION
        {
            public long PerProcessUserTimeLimit;
            public long PerJobUserTimeLimit;
            public uint LimitFlags;
            public UIntPtr MinimumWorkingSetSize;
            public UIntPtr MaximumWorkingSetSize;
            public uint ActiveProcessLimit;
            public UIntPtr Affinity;
            public uint PriorityClass;
            public uint SchedulingClass;
        }

        [StructLayout(LayoutKind.Sequential)]
        private struct IO_COUNTERS
        {
            public ulong ReadOperationCount;
            public ulong WriteOperationCount;
            public ulong OtherOperationCount;
            public ulong ReadTransferCount;
            public ulong WriteTransferCount;
            public ulong OtherTransferCount;
        }

        [StructLayout(LayoutKind.Sequential)]
        private struct JOBOBJECT_EXTENDED_LIMIT_INFORMATION
        {
            public JOBOBJECT_BASIC_LIMIT_INFORMATION BasicLimitInformation;
            public IO_COUNTERS IoInfo;
            public UIntPtr ProcessMemoryLimit;
            public UIntPtr JobMemoryLimit;
            public UIntPtr PeakProcessMemoryUsed;
            public UIntPtr PeakJobMemoryUsed;
        }

        private const int JobObjectExtendedLimitInformation = 9;
        private const uint JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000;

        [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
        private static extern IntPtr CreateJobObject(IntPtr lpJobAttributes, string lpName);

        [DllImport("kernel32.dll", SetLastError = true)]
        private static extern bool SetInformationJobObject(
            IntPtr hJob, int JobObjectInfoClass, IntPtr lpJobObjectInfo, uint cbJobObjectInfoLength);

        [DllImport("kernel32.dll", SetLastError = true)]
        private static extern bool AssignProcessToJobObject(IntPtr hJob, IntPtr hProcess);

        [DllImport("kernel32.dll", SetLastError = true)]
        private static extern bool CloseHandle(IntPtr hObject);

        private IntPtr _handle;

        public JobObject()
        {
            _handle = CreateJobObject(IntPtr.Zero, null);
            if (_handle == IntPtr.Zero)
            {
                throw new InvalidOperationException(
                    "CreateJobObject failed, error " + Marshal.GetLastWin32Error());
            }

            JOBOBJECT_EXTENDED_LIMIT_INFORMATION info = new JOBOBJECT_EXTENDED_LIMIT_INFORMATION();
            info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;

            int length = Marshal.SizeOf(typeof(JOBOBJECT_EXTENDED_LIMIT_INFORMATION));
            IntPtr infoPtr = Marshal.AllocHGlobal(length);
            try
            {
                Marshal.StructureToPtr(info, infoPtr, false);
                if (!SetInformationJobObject(_handle, JobObjectExtendedLimitInformation, infoPtr, (uint)length))
                {
                    int err = Marshal.GetLastWin32Error();
                    CloseHandle(_handle);
                    _handle = IntPtr.Zero;
                    throw new InvalidOperationException("SetInformationJobObject failed, error " + err);
                }
            }
            finally
            {
                Marshal.FreeHGlobal(infoPtr);
            }
        }

        public bool AssignProcess(Process process)
        {
            if (_handle == IntPtr.Zero) return false;
            return AssignProcessToJobObject(_handle, process.Handle);
        }

        public void Dispose()
        {
            if (_handle != IntPtr.Zero)
            {
                // Closing the last handle to a job with
                // JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE set terminates every
                // process still assigned to it -- this is what actually
                // frees the ports when the window is closed.
                CloseHandle(_handle);
                _handle = IntPtr.Zero;
            }
        }
    }

    internal sealed class MainForm : Form
    {
        private readonly Label _statusLabel;
        private readonly TextBox _outputBox;
        private readonly Button _openBrowserButton;
        private readonly Button _stopButton;
        private readonly Timer _readyPollTimer;

        private readonly string _exeDir;
        private readonly string _scriptPath;
        private readonly int _appPort;

        private JobObject _job;
        private Process _child;
        private bool _ready;
        private bool _closingHandled;

        public MainForm()
        {
            _exeDir = GetExeDirectory();
            _scriptPath = Path.Combine(Path.Combine(_exeDir, "app"), "RUN_H3_APP.ps1");
            _appPort = ReadAppPort(Path.Combine(Path.Combine(_exeDir, "app"), "config.json"));

            Text = "H3 Video Maker v2";
            ClientSize = new Size(720, 420);
            MinimumSize = new Size(480, 300);
            StartPosition = FormStartPosition.CenterScreen;
            Icon = SystemIcons.Application;

            _statusLabel = new Label
            {
                Dock = DockStyle.Top,
                Height = 32,
                TextAlign = ContentAlignment.MiddleLeft,
                Padding = new Padding(8, 0, 0, 0),
                Font = new Font(Font.FontFamily, 10f, FontStyle.Bold),
                Text = "起動中..." // 起動中...
            };

            _outputBox = new TextBox
            {
                Dock = DockStyle.Fill,
                Multiline = true,
                ReadOnly = true,
                ScrollBars = ScrollBars.Vertical,
                WordWrap = false,
                Font = new Font(FontFamily.GenericMonospace, 9f)
            };

            FlowLayoutPanel buttonPanel = new FlowLayoutPanel
            {
                Dock = DockStyle.Bottom,
                Height = 44,
                FlowDirection = FlowDirection.RightToLeft,
                Padding = new Padding(8)
            };

            _stopButton = new Button
            {
                Text = "停止して終了", // 停止して終了
                AutoSize = true
            };
            _stopButton.Click += (s, e) => Close();

            _openBrowserButton = new Button
            {
                Text = "ブラウザで開く", // ブラウザで開く
                AutoSize = true
            };
            _openBrowserButton.Click += (s, e) => OpenBrowser();

            buttonPanel.Controls.Add(_stopButton);
            buttonPanel.Controls.Add(_openBrowserButton);

            Controls.Add(buttonPanel);
            Controls.Add(_statusLabel);
            Controls.Add(_outputBox);

            _readyPollTimer = new Timer { Interval = 750 };
            _readyPollTimer.Tick += (s, e) => PollReady();

            Load += MainForm_Load;
            FormClosing += MainForm_FormClosing;
        }

        // ---------------------------------------------------------- launch --
        private void MainForm_Load(object sender, EventArgs e)
        {
            AppendOutput("[launcher] H3 Video Maker v2");
            AppendOutput("[launcher] script: " + _scriptPath);

            if (!File.Exists(_scriptPath))
            {
                SetStatus("エラーで終了しました (コード -1)"); // エラーで終了しました (コード -1)
                AppendOutput("ERROR: launch script not found: " + _scriptPath);
                return;
            }

            try
            {
                _job = new JobObject();
            }
            catch (Exception ex)
            {
                AppendOutput("WARNING: could not create job object: " + ex.Message);
                _job = null;
            }

            ProcessStartInfo startInfo = new ProcessStartInfo
            {
                FileName = "powershell.exe",
                Arguments = "-NoProfile -ExecutionPolicy Bypass -File \"" + _scriptPath + "\"",
                UseShellExecute = false,
                CreateNoWindow = true,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                StandardOutputEncoding = new UTF8Encoding(false),
                StandardErrorEncoding = new UTF8Encoding(false),
                WorkingDirectory = _exeDir
            };

            _child = new Process();
            _child.StartInfo = startInfo;
            _child.EnableRaisingEvents = true;
            _child.OutputDataReceived += (s, args) => AppendOutput(args.Data);
            _child.ErrorDataReceived += (s, args) => AppendOutput(args.Data);
            _child.Exited += Child_Exited;

            try
            {
                _child.Start();
            }
            catch (Exception ex)
            {
                SetStatus("エラーで終了しました (コード -1)");
                AppendOutput("ERROR: failed to start powershell.exe: " + ex.Message);
                return;
            }

            if (_job != null)
            {
                try
                {
                    _job.AssignProcess(_child);
                }
                catch (Exception ex)
                {
                    AppendOutput("WARNING: could not assign process to job object: " + ex.Message);
                }
            }

            _child.BeginOutputReadLine();
            _child.BeginErrorReadLine();

            _readyPollTimer.Start();
        }

        private void Child_Exited(object sender, EventArgs e)
        {
            RunOnUiThread(delegate
            {
                _readyPollTimer.Stop();
                int code = 0;
                try { code = _child.ExitCode; }
                catch { }

                if (code == 0)
                {
                    SetStatus("停止しました"); // 停止しました
                }
                else
                {
                    SetStatus(string.Format(
                        "エラーで終了しました (コード {0})", code));
                    // エラーで終了しました (コード N) -- keep the window open so the
                    // user can read the captured output; do not auto-close.
                }
            });
        }

        // ------------------------------------------------------- readiness --
        // The app's own readiness print ("H3 App: http://127.0.0.1:<port>/")
        // is emitted by a Python process whose stdout is piped through
        // PowerShell into this launcher. Python fully buffers stdout when it
        // is not attached to a console (which is the case here), and the
        // script does not pass -u / set PYTHONUNBUFFERED, so that line can
        // arrive late or be held in the pipe rather than appearing the
        // moment the server actually starts listening. Polling the TCP port
        // directly is not subject to that buffering and reliably reflects
        // the real state, so readiness is detected by polling app_port
        // rather than by scanning captured output for a specific line.
        private void PollReady()
        {
            if (_ready)
            {
                _readyPollTimer.Stop();
                return;
            }

            if (IsPortListening(_appPort))
            {
                _ready = true;
                _readyPollTimer.Stop();
                SetStatus(string.Format(
                    "起動しました: http://127.0.0.1:{0}/", _appPort)); // 起動しました: ...
            }
        }

        private static bool IsPortListening(int port)
        {
            try
            {
                using (TcpClient client = new TcpClient())
                {
                    IAsyncResult result = client.BeginConnect("127.0.0.1", port, null, null);
                    bool signaled = result.AsyncWaitHandle.WaitOne(200);
                    if (!signaled)
                    {
                        return false;
                    }
                    client.EndConnect(result);
                    return true;
                }
            }
            catch
            {
                return false;
            }
        }

        // ----------------------------------------------------------- close --
        private void MainForm_FormClosing(object sender, FormClosingEventArgs e)
        {
            if (_closingHandled) return;
            _closingHandled = true;

            _readyPollTimer.Stop();

            // Close the job handle right away. With
            // JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE set, this kills the
            // powershell child and every descendant it spawned (the app
            // server and, if it launched one, ComfyUI) so the ports are
            // freed. This does not block the UI thread.
            if (_job != null)
            {
                _job.Dispose();
                _job = null;
            }
        }

        // ---------------------------------------------------------- output --
        private void AppendOutput(string line)
        {
            if (line == null) return;
            RunOnUiThread(delegate
            {
                _outputBox.AppendText(line + Environment.NewLine);
            });
        }

        private void SetStatus(string text)
        {
            RunOnUiThread(delegate
            {
                _statusLabel.Text = text;
            });
        }

        private void RunOnUiThread(MethodInvoker action)
        {
            if (IsDisposed) return;
            try
            {
                if (InvokeRequired)
                {
                    BeginInvoke(action);
                }
                else
                {
                    action();
                }
            }
            catch (ObjectDisposedException)
            {
            }
            catch (InvalidOperationException)
            {
                // Handle not created yet or being torn down; safe to ignore.
            }
        }

        private void OpenBrowser()
        {
            try
            {
                Process.Start(new ProcessStartInfo("http://127.0.0.1:" + _appPort + "/")
                {
                    UseShellExecute = true
                });
            }
            catch (Exception ex)
            {
                MessageBox.Show(
                    this,
                    "ブラウザを開けませんでした: " + ex.Message,
                    "H3 Video Maker v2",
                    MessageBoxButtons.OK,
                    MessageBoxIcon.Warning);
            }
        }

        // --------------------------------------------------------- helpers --
        private static string GetExeDirectory()
        {
            string location = Assembly.GetExecutingAssembly().Location;
            return Path.GetDirectoryName(location);
        }

        /// <summary>
        /// Minimal, dependency-free scan for "app_port": N in config.json.
        /// Deliberately not a full JSON parser -- see build constraints.
        /// </summary>
        private static int ReadAppPort(string configPath)
        {
            const int fallback = 8790;
            try
            {
                if (!File.Exists(configPath)) return fallback;
                string text = File.ReadAllText(configPath, Encoding.UTF8);
                Match m = Regex.Match(text, "\"app_port\"\\s*:\\s*(\\d+)");
                if (m.Success)
                {
                    int port;
                    if (int.TryParse(m.Groups[1].Value, out port))
                    {
                        return port;
                    }
                }
            }
            catch
            {
                // Fall through to default.
            }
            return fallback;
        }
    }
}
