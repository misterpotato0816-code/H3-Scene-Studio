const {chromium: playwright} = require('playwright');
const path = require('path');
const {spawn} = require('child_process');
const fs = require('fs');
const assert = require('assert/strict');

(async () => {
  const server = spawn(process.env.H3_TEST_PYTHON || 'python3', [path.join(__dirname,'preview_server.py')], {stdio:'ignore'});
  let browser;
  try {
    for(let i=0;i<50;i++) {
      try { const r = await fetch('http://127.0.0.1:8799/api/heartbeat', {method:'POST',headers:{'Content-Type':'application/json'},body:'{}'}); if(r.ok) break; } catch {}
      await new Promise(r=>setTimeout(r,100));
    }
    browser = await playwright.launch({headless:true, ...(process.env.H3_TEST_CHROMIUM ? {executablePath:process.env.H3_TEST_CHROMIUM} : {}), args:['--no-sandbox','--disable-dev-shm-usage']});
    const page = await browser.newPage({viewport:{width:1440,height:1000}});
    const errors=[];
    page.on('pageerror', e=>errors.push(e.message));
    await page.route('**/api/ai/models*', r=>r.fulfill({json:{ok:true,models:[]}}));
    await page.goto('http://127.0.0.1:8799/');
    const fontRoot=process.env.H3_TEST_FONT_DIR;
    if (fontRoot) {
    await page.route('**/files/*.woff2', r=>r.fulfill({body:fs.readFileSync(fontRoot+'/files/'+r.request().url().split('/').pop()),contentType:'font/woff2'}));
    await page.addStyleTag({content:fs.readFileSync(fontRoot+'/400.css','utf8')+'\nbody,button,input,textarea,select{font-family:"Noto Sans JP",sans-serif!important}'});
    await page.evaluate(()=>document.fonts.ready);
    }
    await page.waitForFunction(()=>state.cfg && document.querySelector('#slots').children.length===4);
    await page.screenshot({path:require('os').tmpdir()+'/h3-desktop.png',fullPage:true});
    assert(await page.locator('#btnNowGo').isVisible());
    assert(await page.locator('#btnNowGo').isDisabled());
    const modes=[['btnModeSingle','cardImages'],['btnModeStory','cardStoryScript'],['btnModeDirector','cardDirector'],['btnModeAudio','cardRehearsal']];
    for(const [button,card] of modes) {
      await page.locator('#'+button).click();
      assert(await page.locator('#'+card).isVisible(),card+' should be visible');
      assert.equal(await page.locator('#'+button).getAttribute('aria-pressed'),'true');
    }
    assert(!(await page.locator('#actionbar').isVisible()),'audio must not show video controls');
    assert.deepEqual(await page.evaluate(()=>Array.from(document.querySelectorAll('#sidenav a')).filter(a=>!document.querySelector(a.getAttribute('href'))).map(a=>a.href)),[]);
    let released=false;
    await page.route('**/api/release-models', r=>{
      assert.equal(r.request().headers()['content-type'],'application/json');released=true;
      return r.fulfill({json:{ok:true,message:'fixture release'}});
    });
    await page.locator('#btnRelease').click();
    await page.waitForFunction(()=>document.getElementById('releaseStatus').textContent.includes('fixture'));
    assert(released);
    await page.route('**/api/voice-rehearsal/speakers', r=>r.fulfill({json:{ok:true,speakers:[{name:'テスト話者',styles:[{name:'標準',id:0}]}]}}));
    await page.route('**/api/voice-rehearsal/query', r=>r.fulfill({json:{ok:true,query:{accent_phrases:[{accent:1,moras:[{text:'ア'},{text:'メ'}]}]}}}));
    await page.locator('#btnVoiceConnect').click();
    await page.locator('#voiceSpeaker option').waitFor({state:'attached'});
    await page.locator('#rehearsalText').fill('雨、降りそうだね。');
    await page.locator('#btnVoiceAnalyze').click();
    await page.locator('#voiceAccents select').waitFor();
    assert(await page.locator('#btnVoicePlay').isEnabled());
    await page.locator('#rehearsalText').fill('今日は晴れ。');
    assert(await page.locator('#btnVoicePlay').isDisabled());
    await page.locator('#btnModeDirector').click();
    await page.evaluate(()=>{
      const names=['video_type','location','scenery','subject','outfit','acting','camera_style','camera_work','dialogue','voice','ambient_audio','mood'];
      state.director.spec={kind:'single',duration_sec:5,items:Object.fromEntries(names.map(n=>[n,{value:n==='dialogue'?'あ、見て。きれいだね。':'静かな夕暮れ',en:'A quiet evening.',locked:false,request:''}])),timeline:[{t0:0,t1:5,label:'目線を上げて、ひとこと。',en:'A person looks up.'}],cast:{on_screen_subjects:['character:0']}};
      renderDirectorResult();setMode('director');
    });
    const dialogue=page.locator('.dialogue-edit');
    await dialogue.fill('あれ、雨かな？');
    assert.equal(await page.evaluate(()=>state.director.spec.items.dialogue.value),'あれ、雨かな？');
    const times=page.locator('.timeline-row input');
    await times.nth(1).fill('8');
    await page.locator('.direction-review summary').click();
    await page.locator('#btnDirectionReview').click();
    await page.waitForFunction(()=>document.getElementById('directionReview').textContent.includes('ショット1'));
    await times.nth(1).fill('5');
    await page.locator('#btnDirectionReview').click();
    await page.waitForFunction(()=>document.getElementById('directionReview').textContent.includes('あれ、雨かな？'));
    await page.screenshot({path:require('os').tmpdir()+'/h3-director.png',fullPage:true});
    for(const width of [900,390]) {
      await page.setViewportSize({width,height:900});
      for(const [button] of modes) {
        await page.locator('#'+button).click();
        const size=await page.evaluate(()=>({scroll:document.documentElement.scrollWidth,client:document.documentElement.clientWidth}));
        assert(size.scroll<=size.client,`${width}px ${button}: overflow ${JSON.stringify(size)}`);
      }
    }
    await page.locator('#btnModeSingle').click();
    await page.screenshot({path:require('os').tmpdir()+'/h3-mobile.png',fullPage:true});
    assert.deepEqual(errors,[]);
    console.log('PASS: boot, 4 modes, editable dialogue, timeline review, voice query invalidation, 900/390px overflow, no JS errors.');
  } finally { if(browser) await browser.close(); server.kill('SIGTERM'); }
})().catch(e=>{console.error(e);process.exitCode=1;});

