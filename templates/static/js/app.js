let isStreaming=false,stream=null,intervalId=null;
let lipColor='#FF0000',blushColor='#FFB7C5',eyeColor='#B46482';
let opacity=0.6,lipstickOn=true,blushOn=false,eyeshadowOn=false;
let lastTone='';

document.addEventListener('DOMContentLoaded',()=>{
  const video=document.getElementById('video');
  const img=document.getElementById('processedImg');
  const btn=document.getElementById('startBtn');
  const ph=document.getElementById('ph');

  btn.addEventListener('click',async()=>{ isStreaming?stop():await start(); });

  async function start(){
    try{
      stream=await navigator.mediaDevices.getUserMedia({video:{width:320,height:240,facingMode:'user'},audio:false});
      video.srcObject=stream;
      await video.play();
      isStreaming=true;
      btn.textContent='⬛  Stop Camera';
      if(ph)ph.style.display='none';
      intervalId=setInterval(send,150);
    }catch(e){
      if(e.name==='NotAllowedError') alert('Camera blocked!\nClick camera icon in address bar → Allow → Refresh.');
      else alert('Camera error: '+e.message);
    }
  }

  function stop(){
    if(stream)stream.getTracks().forEach(t=>t.stop());
    stream=null; video.srcObject=null; isStreaming=false;
    btn.textContent='⬤  Start Camera';
    if(ph)ph.style.display='block';
    img.style.display='none'; video.style.display='block';
    clearInterval(intervalId); intervalId=null;
  }

  async function send(){
    if(!isStreaming||!video.videoWidth)return;
    try{
      const c=document.createElement('canvas');
      c.width=video.videoWidth; c.height=video.videoHeight;
      const x=c.getContext('2d');
      x.translate(c.width,0); x.scale(-1,1);
      x.drawImage(video,0,0);
      const b64=c.toDataURL('image/jpeg',0.4).split(',')[1];
      const res=await fetch('/process_frame',{
        method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({image:b64,settings:{lip_color:lipColor,blush_color:blushColor,eye_color:eyeColor,opacity,lipstick:lipstickOn,blush:blushOn,eyeshadow:eyeshadowOn}})
      });
      if(!res.ok)return;
      const d=await res.json();
      if(d.processed_image){img.src='data:image/jpeg;base64,'+d.processed_image;img.style.display='block';video.style.display='none';}
      if(d.skin_tone){const el=document.getElementById('skinTone');if(el)el.textContent=d.skin_tone;}
      if(d.pipeline)updatePipeline(d.pipeline);
      if(d.face_detected&&d.skin_tone)loadRecs(d.skin_tone);
    }catch(e){console.error(e);}
  }

  function updatePipeline(p){
    const map={faceDetection:'step-face',landmarkExtraction:'step-landmark',skinToneClassification:'step-skin',makeupApplication:'step-makeup',recommendationEngine:'step-recs'};
    Object.entries(map).forEach(([k,id])=>{
      const el=document.getElementById(id); if(!el)return;
      const st=el.querySelector('.sstatus');
      if(p[k]){el.className='ps complete';if(st)st.textContent='Complete ✓';}
      else if(k==='faceDetection'){el.className='ps running';if(st)st.textContent='Running...';}
    });
  }

  async function loadRecs(tone){
    if(tone===lastTone)return; lastTone=tone;
    try{
      const d=await(await fetch('/recommendations')).json();
      const recs=d[tone]||d['Unknown']||[];
      const box=document.getElementById('recommendations');
      if(box)box.innerHTML=recs.map(r=>`
        <div class="ri">
          <div class="rd" style="background:${r.shade}"></div>
          <div class="rinfo"><div class="rname">${r.name}</div><div class="rtype">${r.type}</div></div>
          <button class="rtry" style="background:${r.shade}" onclick="applyShade('${r.shade}')">Try On</button>
        </div>`).join('');
    }catch(e){}
  }

  window.applyShade=s=>{ lipColor=s; document.querySelectorAll('#lipS .swatch').forEach(x=>x.classList.remove('active')); };

  function swatches(cid,fn){
    const c=document.getElementById(cid); if(!c)return;
    c.querySelectorAll('.swatch').forEach(s=>s.addEventListener('click',()=>{
      c.querySelectorAll('.swatch').forEach(x=>x.classList.remove('active'));
      s.classList.add('active'); fn(s.getAttribute('data-color'));
    }));
  }
  swatches('lipS',  c=>lipColor=c);
  swatches('blushS',c=>blushColor=c);
  swatches('eyeS',  c=>eyeColor=c);

  const sl=document.getElementById('opacitySlider');
  if(sl)sl.addEventListener('input',e=>{opacity=e.target.value/100;document.getElementById('opacityValue').textContent=e.target.value+'%';});

  function tog(id,fn,sid){
    const el=document.getElementById(id),sec=document.getElementById(sid); if(!el)return;
    el.addEventListener('change',e=>{fn(e.target.checked);if(sec)sec.classList.toggle('disabled',!e.target.checked);});
  }
  tog('lipstickToggle', v=>lipstickOn=v,  'lipCS');
  tog('blushToggle',    v=>blushOn=v,     'blushCS');
  tog('eyeshadowToggle',v=>eyeshadowOn=v, 'eyeCS');
});
