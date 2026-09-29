    const { useState, useEffect, useRef, useLayoutEffect, isValidElement } = React;

    const Ico = ({ children, size=16, color="currentColor" }) => (
      <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">{children}</svg>
    );
    const CameraIcon   = () => <Ico><path d="M14.5 4h-5L7 7H4a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V9a2 2 0 0 0-2-2h-3l-2.5-3z"/><circle cx="12" cy="13" r="3"/></Ico>;
    const RadarIcon    = () => <Ico><path d="M12 22a10 10 0 1 0 0-20 10 10 0 0 0 0 20z"/><path d="M12 12 7.5 7.5"/><path d="M12 12 16.5 7.5"/></Ico>;
    const LayersIcon   = () => <Ico><polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 12 12 17 22 12"/><polyline points="2 17 12 22 22 17"/></Ico>;
    const AlertIcon    = () => <Ico><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></Ico>;
    const ActivityIcon = () => <Ico><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></Ico>;
    const DatabaseIcon = () => <Ico><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/></Ico>;
    const SettingsIcon = () => <Ico><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"/></Ico>;
    const DownloadIcon = () => <Ico><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></Ico>;
    const ShieldIcon   = ({ size=28, color="#78A578" }) => <Ico size={size} color={color}><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></Ico>;
    const MenuIcon     = () => <Ico size={20}><line x1="3" y1="6" x2="21" y2="6"/><line x1="3" y1="12" x2="21" y2="12"/><line x1="3" y1="18" x2="21" y2="18"/></Ico>;

    // BranchedMenu
    const PAD=6,MARK=16;
    const renderIconItem=icon=>(isValidElement(icon)?icon:null);
    const toSet=open=>new Set(Array.isArray(open)?open:open>=0?[open]:[]);

    function BranchedMenu({items,defaultOpen=0,defaultActive='',onSelect,onToggle,
      color='#f5f5f5',accentColor='#f5f5f5',lineColor='#3f3f46',
      width=240,rowHeight=36,indent=40,trunk=14,radius=10,
      lineWidth=1.5,fontSize=14,drawDuration=400,foldDuration=300,className=''}) {
      const [open,setOpen]=useState(()=>toSet(defaultOpen));
      const [active,setActive]=useState(()=>{
        if(defaultActive)return defaultActive;
        const first=items.find((it,i)=>it.children&&toSet(defaultOpen).has(i));
        return first?.children?.[0]?.value??'';
      });
      const navRef=useRef(null),heads=useRef([]),markerRef=useRef(null),latest=useRef({});
      latest.current={onSelect,onToggle};
      const activeSection=items.findIndex(it=>it.children?.some(kid=>kid.value===active));
      const markerShown=activeSection>=0&&open.has(activeSection);
      useLayoutEffect(()=>{
        const place=glide=>{
          const m=markerRef.current,el=heads.current[activeSection];
          if(!m)return;const on=markerShown&&el;
          if(!glide)m.style.transition='none';
          if(on)m.style.top=`${el.offsetTop+(el.offsetHeight-MARK)/2}px`;
          if(on)m.setAttribute('data-on','true');else m.removeAttribute('data-on');
          if(!glide){void m.offsetHeight;m.style.transition='';}
        };
        place(true);let first=true;
        const hr=()=>{if(first){first=false;return;}place(false);};
        window.addEventListener('resize',hr);return()=>window.removeEventListener('resize',hr);
      },[activeSection,markerShown,items,fontSize,rowHeight]);
      const select=(value,item)=>{setActive(value);if(latest.current.onSelect)latest.current.onSelect(value,item);};
      const toggle=i=>{setOpen(prev=>{const next=new Set(prev);if(!next.has(i))next.add(i);else next.delete(i);if(latest.current.onToggle)latest.current.onToggle(i,!prev.has(i));return next;});};
      const r=Math.min(radius,rowHeight/2-2),endX=indent-8;
      const rowY=k=>PAD+k*rowHeight+rowHeight/2;
      const branch=k=>`M ${trunk} ${rowY(k)-r} A ${r} ${r} 0 0 0 ${trunk+r} ${rowY(k)} H ${endX}`;
      const reach=k=>`M ${trunk} 0 V ${rowY(k)-r} A ${r} ${r} 0 0 0 ${trunk+r} ${rowY(k)} H ${endX}`;
      const length=k=>rowY(k)-r+(Math.PI*r)/2+(endX-trunk-r);
      return (
        <nav ref={navRef} className={`branched-menu${className?` ${className}`:''}`}
          style={{'--bm-w':`${width}px`,'--bm-ink':color,'--bm-accent':accentColor,'--bm-line':lineColor,'--bm-font':`${fontSize}px`,'--bm-row':`${rowHeight}px`,'--bm-indent':`${indent}px`,'--bm-line-w':lineWidth,'--bm-draw':`${drawDuration}ms`,'--bm-fold':`${foldDuration}ms`}}>
          <span ref={markerRef} className="branched-menu__marker" aria-hidden="true"/>
          {items.map((item,i)=>{
            const kids=item.children,isOpen=kids?open.has(i):false;
            const leafValue=item.value??item.label,leafActive=!kids&&leafValue===active;
            const bodyH=kids?PAD*2+kids.length*rowHeight:0;
            return (
              <div key={item.value??item.label} className="branched-menu__section" data-open={isOpen?'':undefined}>
                <button ref={el=>{heads.current[i]=el;}} type="button" className="branched-menu__head"
                  aria-expanded={kids?isOpen:undefined} aria-current={leafActive?'true':undefined}
                  data-active={leafActive?'':undefined} onClick={()=>(kids?toggle(i):select(leafValue,item))}>
                  {item.label}
                </button>
                {kids?(
                  <div className="branched-menu__body"><div className="branched-menu__fold">
                    <div className="branched-menu__tree" style={{height:bodyH}}>
                      <svg className="branched-menu__lines" width={indent} height={bodyH} aria-hidden="true">
                        <path className="branched-menu__base" d={`M ${trunk} 0 V ${rowY(kids.length-1)-r}`}/>
                        {kids.map((kid,k)=><path key={kid.value} className="branched-menu__base" d={branch(k)}/>)}
                        {kids.map((kid,k)=>(<path key={kid.value} className="branched-menu__reach" d={reach(k)} style={{strokeDasharray:length(k),strokeDashoffset:kid.value===active?0:length(k)}}/>))}
                      </svg>
                      {kids.map(kid=>(
                        <button key={kid.value} type="button" className="branched-menu__item"
                          aria-current={kid.value===active?'true':undefined} data-active={kid.value===active?'':undefined}
                          tabIndex={isOpen?0:-1} onClick={()=>select(kid.value,kid)}>
                          {kid.icon?<span className="branched-menu__icon" aria-hidden="true">{renderIconItem(kid.icon)}</span>:null}
                          <span className="branched-menu__label">{kid.label}</span>
                        </button>
                      ))}
                    </div>
                  </div></div>
                ):null}
              </div>
            );
          })}
        </nav>
      );
    }

    function TypeWriter({ texts, speed=60, pause=2200 }) {
      const [display, setDisplay] = useState('');
      const [tIdx, setTIdx] = useState(0);
      const [cIdx, setCIdx] = useState(0);
      const [deleting, setDeleting] = useState(false);
      useEffect(() => {
        const current = texts[tIdx];
        const delay = deleting ? speed / 2 : speed;
        const t = setTimeout(() => {
          if (!deleting && cIdx < current.length) {
            setDisplay(current.slice(0, cIdx + 1));
            setCIdx(c => c + 1);
          } else if (!deleting && cIdx === current.length) {
            setTimeout(() => setDeleting(true), pause);
          } else if (deleting && cIdx > 0) {
            setDisplay(current.slice(0, cIdx - 1));
            setCIdx(c => c - 1);
          } else {
            setDeleting(false);
            setTIdx(i => (i + 1) % texts.length);
          }
        }, delay);
        return () => clearTimeout(t);
      }, [cIdx, deleting, tIdx]);
      return <span>{display}<span className="cursor-blink">|</span></span>;
    }

    function LandingPage({onEnter, onPrivacy, onTerms}){
      useEffect(()=>{
        if(window.setAuroraState)window.setAuroraState(1,1.4,0.65,0.35);
      },[]);
      return (
        <div className="landing-page">
          <div className="landing-overlay"></div>
          <div className="landing-content">
            <div className="landing-logo-ring">
              <ShieldIcon size={64} color="#FF9933"/>
            </div>
            <h1 className="landing-title">SEEMA-DRISHTI</h1>
            <p className="landing-typewriter">
              <TypeWriter texts={['Border Surveillance System','Threat Intelligence Platform','Real-Time Radar Network','Perimeter Defense Grid']} speed={55} pause={2000}/>
            </p>
            <div className="landing-actions">
              <button className="landing-btn" id="btn-access-terminal" onClick={onEnter}>ACCESS TERMINAL</button>
              <div className="landing-clearance">AUTHORIZED PERSONNEL ONLY / MINISTRY OF DEFENCE</div>
            </div>
          </div>
          <div className="landing-footer">
            <button className="landing-footer-link" onClick={onPrivacy}>Privacy Policy</button>
            <span className="landing-footer-sep">|</span>
            <button className="landing-footer-link" onClick={onTerms}>Terms of Use</button>
          </div>
        </div>
      );
    }

    function PrivacyPage({onBack}){
      return (
        <div className="legal-page">
          <div className="legal-overlay"></div>
          <div className="legal-content">
            <button className="legal-back" onClick={onBack}>Back</button>
            <h1>Privacy Policy</h1>
            <p className="legal-effective">Effective Date: 27 September 2026</p>
            <section>
              <h2>1. Information We Collect</h2>
              <p>SEEMA-DRISHTI collects only the information necessary to authenticate authorized personnel and operate the surveillance platform. This includes operative credentials entered at login, session identifiers, and system access logs.</p>
            </section>
            <section>
              <h2>2. How We Use Information</h2>
              <p>All collected data is used exclusively for authentication, audit trail maintenance, and platform security. No data is shared with third parties outside the Ministry of Defence chain of command.</p>
            </section>
            <section>
              <h2>3. Data Retention</h2>
              <p>Session and access log data is retained for 90 days in compliance with government information security directives, after which it is securely deleted.</p>
            </section>
            <section>
              <h2>4. Security</h2>
              <p>All data is encrypted in transit and at rest using AES-256 encryption. Access to this platform is restricted to personnel with verified security clearance.</p>
            </section>
            <section>
              <h2>5. Contact</h2>
              <p>For privacy-related inquiries, contact the SEEMA-DRISHTI system administrator through the Ministry of Defence secure communications channel.</p>
            </section>
          </div>
        </div>
      );
    }

    function TermsPage({onBack}){
      return (
        <div className="legal-page">
          <div className="legal-overlay"></div>
          <div className="legal-content">
            <button className="legal-back" onClick={onBack}>Back</button>
            <h1>Terms of Use</h1>
            <p className="legal-effective">Effective Date: 27 September 2026</p>
            <section>
              <h2>1. Authorized Use Only</h2>
              <p>SEEMA-DRISHTI is a classified platform operated under the authority of the Ministry of Defence, Government of India. Access is restricted to personnel who have been granted explicit security clearance. Unauthorized access is a criminal offence under the Information Technology Act, 2000, and the Official Secrets Act, 1923.</p>
            </section>
            <section>
              <h2>2. User Obligations</h2>
              <p>Authorized users must not share credentials, access the platform from unsecured networks, or export any intelligence data without proper authorization. All activity on this platform is logged and subject to audit.</p>
            </section>
            <section>
              <h2>3. Prohibited Actions</h2>
              <p>Users must not attempt to reverse engineer, tamper with, or circumvent any security mechanism on this platform. Any such attempt will result in immediate credential revocation and referral for prosecution.</p>
            </section>
            <section>
              <h2>4. Disclaimer</h2>
              <p>The Ministry of Defence provides this platform on an as-is basis. While all reasonable measures are taken to ensure accuracy of surveillance data, the platform does not guarantee the completeness of any intelligence output.</p>
            </section>
            <section>
              <h2>5. Governing Law</h2>
              <p>These terms are governed by the laws of the Republic of India. Any disputes shall be subject to the exclusive jurisdiction of courts in New Delhi.</p>
            </section>
          </div>
        </div>
      );
    }

    function SignInPage({onSignIn}){
      const [scanning,setScanning]=useState(false);
      const [scanText,setScanText]=useState("AWAITING CREDENTIALS...");
      const [username,setUsername]=useState("");
      const [password,setPassword]=useState("");
      const [error,setError]=useState("");
      useEffect(()=>{if(window.setAuroraState)window.setAuroraState(1,0.8,0.7,0.2);},[]);
      const handleAuth=()=>{
        if(!username.trim()||!password.trim()){setError("WARNING: CREDENTIALS REQUIRED");setTimeout(()=>setError(""),2500);return;}
        setError("");setScanning(true);setScanText("VERIFYING BIOMETRICS...");
        setTimeout(()=>setScanText("DECRYPTING CLEARANCE..."),1000);
        setTimeout(()=>setScanText("ACCESS GRANTED"),2000);
        setTimeout(()=>onSignIn(),2600);
      };
      const handleKey=e=>{if(e.key==='Enter'&&!scanning)handleAuth();};
      return (
        <div className="login-wrapper">
          <div className="landing-overlay"></div>
          <div className="login-card">
            <div className="login-header">
              <ShieldIcon size={56} color="#FF9933"/>
              <h2>RESTRICTED ACCESS</h2>
              <div className={`status-text${scanning?' pulse':''}`}>{scanText}</div>
            </div>
            <div className="input-group">
              <label htmlFor="operative-id">OPERATIVE ID</label>
              <input id="operative-id" type="text" placeholder="e.g. ALPHA-77"
                value={username} onChange={e=>setUsername(e.target.value)}
                onKeyDown={handleKey} disabled={scanning} autoComplete="username"/>
            </div>
            <div className="input-group">
              <label htmlFor="clearance-code">CLEARANCE CODE</label>
              <input id="clearance-code" type="password" placeholder="Enter clearance code"
                value={password} onChange={e=>setPassword(e.target.value)}
                onKeyDown={handleKey} disabled={scanning} autoComplete="current-password"/>
            </div>
            {error&&<div className="error-msg">{error}</div>}
            {scanning&&(
              <div className="auth-scanning-bar">
                <div className="auth-scanning-fill"></div>
              </div>
            )}
            <button id="btn-initiate-handshake" className="auth-btn" onClick={handleAuth} disabled={scanning}>
              {scanning?'AUTHENTICATING...':'INITIATE HANDSHAKE'}
            </button>
            <p className="login-hint">Press Enter to authenticate</p>
          </div>
        </div>
      );
    }

    /* ── FlipCard: dark/light mode toggle ── */
    function FlipCard({isFlipped, onFlip}){
      const cardRef=useRef(null);
      const [tilt,setTilt]=useState({x:0,y:0});
      const [glare,setGlare]=useState({x:50,y:50});
      const [hovered,setHovered]=useState(false);
      const onMove=e=>{
        if(!cardRef.current)return;
        const r=cardRef.current.getBoundingClientRect();
        const cx=r.left+r.width/2, cy=r.top+r.height/2;
        const tx=((e.clientY-cy)/(r.height/2))*-10;
        const ty=((e.clientX-cx)/(r.width/2))*10;
        setTilt({x:tx,y:ty});
        setGlare({x:((e.clientX-r.left)/r.width)*100,y:((e.clientY-r.top)/r.height)*100});
      };
      const tiltStyle=hovered
        ?{transform:`perspective(1100px) rotateX(${tilt.x}deg) rotateY(${tilt.y}deg) scale(1.06)`,transition:'transform 0.08s linear'}
        :{transform:'perspective(1100px) rotateX(0deg) rotateY(0deg) scale(1)',transition:'transform 0.45s cubic-bezier(0.23,1,0.32,1)'};
      return (
        <div ref={cardRef} className="fc-wrap" onClick={onFlip}
          onMouseMove={onMove}
          onMouseEnter={()=>setHovered(true)}
          onMouseLeave={()=>{setHovered(false);setTilt({x:0,y:0});}}
          style={tiltStyle}>
          <div className={`fc-inner${isFlipped?' fc-flipped':''}`}>
            <div className="fc-face fc-front">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>
              <span>DARK</span>
              {hovered&&<div className="fc-glare" style={{background:`radial-gradient(ellipse at ${glare.x}% ${glare.y}%,rgba(255,255,255,0.18) 0%,transparent 65%)`}}/>}
            </div>
            <div className="fc-face fc-back">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/></svg>
              <span>LIGHT</span>
              {hovered&&<div className="fc-glare" style={{background:`radial-gradient(ellipse at ${glare.x}% ${glare.y}%,rgba(255,255,255,0.22) 0%,transparent 65%)`}}/>}
            </div>
          </div>
        </div>
      );
    }


    function Dashboard({time}){
      const [sidebarOpen,setSidebarOpen]=useState(false);
      const [isDark,setIsDark]=useState(true);
      const toggleMode=()=>{
        const next=!isDark;
        setIsDark(next);
        if(window.setBgMode)window.setBgMode(next?'dark':'light');
      };
      const [activeView,setActiveView]=useState('command-center');
      const closeTimer=useRef(null);
      const videoRef=useRef(null);
      const canvasRef=useRef(null);
      const lastFrameRef=useRef(null);
      const frozenCountRef=useRef(0);
      const [camActive,setCamActive]=useState(false);
      const [incidentSimulated,setIncidentSimulated]=useState(false);
      const [tamperStatus,setTamperStatus]=useState('NOMINAL');
      const [freezeDetected,setFreezeDetected]=useState(false);
      const [fourEyesPending,setFourEyesPending]=useState(false);
      const [fourEyesCountdown,setFourEyesCountdown]=useState(120);
      const [linkStatus,setLinkStatus]=useState('ONLINE');
      const [queuedEvents,setQueuedEvents]=useState(0);
      const countdownTimer=useRef(null);
      const mapRef=useRef(null);
      const mapInst=useRef(null);
      const [userCoords,setUserCoords]=useState(null);
      const handleLocate=()=>{
        if(!navigator.geolocation){
          alert('Location requires HTTPS or localhost. Access this page at http://localhost:8080 on this device, or enable HTTPS.');
          return;
        }
        navigator.geolocation.getCurrentPosition(
          pos=>{
            const lat=pos.coords.latitude.toFixed(4);
            const lng=pos.coords.longitude.toFixed(4);
            setUserCoords(lat+'N '+lng+'E');
            if(mapInst.current){
              const L=window.L;
              mapInst.current.setView([parseFloat(lat),parseFloat(lng)],12);
              const icon=L.divIcon({className:'',html:'<div style="width:16px;height:16px;border-radius:50%;background:#4a90ff;border:3px solid #fff;box-shadow:0 0 12px #4a90ff"></div>',iconSize:[16,16],iconAnchor:[8,8]});
              L.marker([parseFloat(lat),parseFloat(lng)],{icon}).addTo(mapInst.current).bindPopup('Your Position').openPopup();
            }
          },
          err=>alert('Location denied: '+err.message),
          {enableHighAccuracy:true,timeout:8000}
        );
      };
      useEffect(()=>{
        if(activeView!=='border-map')return;
        const L=window.L;
        if(!L||!mapRef.current)return;
        if(mapInst.current){setTimeout(()=>mapInst.current.invalidateSize(),150);return;}
        const map=L.map(mapRef.current,{zoomControl:true,attributionControl:true});
        L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',{maxZoom:18,attribution:'Tiles Esri'}).addTo(map);
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',{maxZoom:18,opacity:0.28}).addTo(map);
        map.setView([30.25,74.0],7);
        const cams=[
          {id:'CAM-07',lat:28.263,lng:72.553,s:'online'},
          {id:'CAM-08',lat:28.242,lng:72.568,s:'incident'},
          {id:'CAM-09',lat:28.273,lng:72.588,s:'online'},
          {id:'CAM-12',lat:28.232,lng:72.544,s:'degraded'},
          {id:'CAM-03',lat:28.284,lng:72.608,s:'online'},
          {id:'BOP-07',lat:28.236,lng:72.598,s:'online'},
        ];
        const colors={online:'#78A578',incident:'#ff4a4a',degraded:'#e8a838'};
        cams.forEach(cam=>{
          const col=colors[cam.s]||'#888';
          const icon=L.divIcon({className:'',html:`<div style='width:12px;height:12px;border-radius:50%;background:${col};border:2px solid #fff;box-shadow:0 0 8px ${col}'></div>`,iconSize:[12,12],iconAnchor:[6,6]});
          L.marker([cam.lat,cam.lng],{icon}).addTo(map).bindPopup(`<b>${cam.id}</b><br>${cam.s.toUpperCase()}`);
        });
        mapInst.current=map;
        setTimeout(()=>map.invalidateSize(),250);
      },[activeView]);

      const today=new Date();
      const dateStr=today.toLocaleDateString('en-GB',{day:'2-digit',month:'short',year:'numeric'}).toUpperCase().replace(/ /g,' ');
      const [sidebarItems,setSidebarItems]=useState([
        {id:'command-center',label:'Command Center'},
        {id:'live-surveillance',label:'Live Surveillance'},
        {id:'border-map',label:'Border Map'},
        {id:'incidents',label:'Incidents',badge:3},
        {id:'evidence',label:'Evidence'},
        {id:'analytics',label:'Analytics'},
        {id:'camera-health',label:'Camera Health'},
        {id:'audit-trail',label:'Audit Trail'},
        {id:'account',label:'Account'},
      ]);
      const [stats,setStats]=useState([
        {label:'CAMERAS',val:'24',sub:'23 online'},
        {label:'INCIDENTS',val:'03',sub:'2 high priority'},
        {label:'EVENTS (24H)',val:'1,284',sub:'up 12%',subColor:'#78A578'},
        {label:'SYSTEM HEALTH',val:'97%',sub:'1 camera degraded',subColor:'#c0a050'},
      ]);
      const [cameras,setCameras]=useState([
        {id:'CAM-07',x:28,y:52,status:'online'},
        {id:'CAM-08',x:48,y:65,status:'incident'},
        {id:'CAM-09',x:68,y:38,status:'online'},
        {id:'BOP-07',x:72,y:72,status:'online'},
      ]);
      const [incidents,setIncidents]=useState([
        {id:'#1047',desc:'Possible perimeter breach',cameras:'CAM-07 > CAM-08 > CAM-09',time:'14:31:04',ai:91,ev:86,severity:'high',intent:'CROSSING',awaitingConfirm:true,cam07t:true},
        {id:'#1046',desc:'Restricted zone - repeated probing',cameras:'CAM-12',time:'14:28:04',ai:78,ev:74,severity:'medium',intent:'LOITERING',awaitingConfirm:false,cam07t:false},
        {id:'#1045',desc:'Unattended object near fence',cameras:'CAM-03',time:'13:52:11',ai:64,ev:58,severity:'low',intent:'STATIC',awaitingConfirm:false,cam07t:false},
      ]);
      const [reidTracks]=useState([
        {id:'ALPHA-7',color:'#e8a838',path:['CAM-07 14:28','CAM-08 14:31','CAM-09 14:33'],signature:'Gait+Clothing (Blue vest, 1.78m, stride 0.72m)',note:'No facial data used'},
        {id:'BETA-3',color:'#5890c8',path:['CAM-03 13:49','CAM-07 13:55'],signature:'Motion Vector + Silhouette',note:'Re-identified via posture profile'},
      ]);
      const [fpStats]=useState([
        {cam:'CAM-09',total:142,fp:97,pct:68,breakdown:[{label:'Livestock at dusk',pct:68},{label:'Foliage movement',pct:21},{label:'Weather artifact',pct:11}]},
        {cam:'CAM-03',total:88,fp:40,pct:45,breakdown:[{label:'Weather artifacts',pct:45},{label:'Bird/wildlife',pct:32},{label:'Lens flare',pct:23}]},
        {cam:'CAM-07',total:210,fp:21,pct:10,breakdown:[{label:'Shadow patterns',pct:52},{label:'Civilian (authorized)',pct:48}]},
      ]);
      const camHealthData=[
        {id:'CAM-07',signal:98,tamper:'NOMINAL',freeze:'CLEAR',lastVerified:'27 SEP 2026 14:00',status:'online'},
        {id:'CAM-08',signal:0,tamper:'SIGNAL LOST',freeze:'N/A',lastVerified:'27 SEP 2026 13:45',status:'incident'},
        {id:'CAM-09',signal:91,tamper:'NOMINAL',freeze:'CLEAR',lastVerified:'27 SEP 2026 14:02',status:'online'},
        {id:'CAM-12',signal:73,tamper:'DEGRADED',freeze:'CLEAR',lastVerified:'27 SEP 2026 14:01',status:'degraded'},
        {id:'CAM-03',signal:87,tamper:'NOMINAL',freeze:'CLEAR',lastVerified:'27 SEP 2026 13:58',status:'online'},
        {id:'BOP-07',signal:99,tamper:'NOMINAL',freeze:'CLEAR',lastVerified:'27 SEP 2026 14:03',status:'online'},
      ];
      const severityColor={high:'#ff4a4a',medium:'#e8a838',low:'#5890c8'};
      const intentColor={CROSSING:'#ff4a4a',LOITERING:'#e8a838',STATIC:'#5890c8','RECON-PATTERN':'#c85890'};
      const startCamera=async()=>{
        if(!navigator.mediaDevices||!navigator.mediaDevices.getUserMedia){
          alert('Camera requires HTTPS or localhost. Access this page at http://localhost:8080 on this device, or enable HTTPS.');
          return;
        }
        try{
          const stream=await navigator.mediaDevices.getUserMedia({video:{facingMode:'environment',width:{ideal:1280},height:{ideal:720}},audio:false});
          if(videoRef.current){videoRef.current.srcObject=stream;videoRef.current.play();}
          setCamActive(true);
        }catch(e){
          if(e.name==='NotAllowedError')alert('Camera permission denied. Please allow camera access in browser settings.');
          else if(e.name==='NotFoundError')alert('No camera found on this device.');
          else alert('Camera error: '+e.message);
        }
      };
      useEffect(()=>{
        if(!camActive)return;
        const canvas=document.createElement('canvas');
        canvas.width=80;canvas.height=45;
        canvasRef.current=canvas;
        const ctx=canvas.getContext('2d');
        const interval=setInterval(()=>{
          const vid=videoRef.current;
          if(!vid||vid.readyState<2)return;
          ctx.drawImage(vid,0,0,80,45);
          const frame=ctx.getImageData(0,0,80,45).data;
          if(lastFrameRef.current){
            let diff=0,dark=0,total=frame.length/4;
            for(let i=0;i<frame.length;i+=4){
              diff+=Math.abs(frame[i]-lastFrameRef.current[i])+Math.abs(frame[i+1]-lastFrameRef.current[i+1])+Math.abs(frame[i+2]-lastFrameRef.current[i+2]);
              if((frame[i]+frame[i+1]+frame[i+2])/3<12)dark++;
            }
            const avgDiff=diff/(total*3);
            const darkRatio=dark/total;
            if(darkRatio>0.72){setTamperStatus('BLACKOUT / TAMPER SUSPECTED');}
            else if(avgDiff>90){setTamperStatus('ANGLE DRIFT DETECTED');}
            else{setTamperStatus('NOMINAL');}
            if(avgDiff<0.8){frozenCountRef.current++;if(frozenCountRef.current>6)setFreezeDetected(true);}
            else{frozenCountRef.current=0;setFreezeDetected(false);}
          }
          lastFrameRef.current=new Uint8Array(frame);
        },500);
        return()=>clearInterval(interval);
      },[camActive]);
      const simulateIncident=()=>{
        setIncidentSimulated(true);
        setCameras(prev=>prev.map(c=>c.id==='CAM-07'?{...c,status:'incident'}:c));
        setStats(prev=>{let n=[...prev];n[1]={...n[1],val:'04',sub:'3 high priority'};return n;});
        setSidebarItems(prev=>{let n=[...prev];n[3]={...n[3],badge:4};return n;});
        const t=new Date().toLocaleTimeString('en-GB',{hour12:false});
        setIncidents(prev=>[{id:'#1048',desc:'Unauthorized movement on live feed',cameras:'CAM-07',time:t,ai:98,ev:95,severity:'high',intent:'CROSSING',awaitingConfirm:true,cam07t:true},...prev]);
        setFourEyesPending(true);
        setFourEyesCountdown(120);
        if(countdownTimer.current)clearInterval(countdownTimer.current);
        countdownTimer.current=setInterval(()=>{setFourEyesCountdown(s=>{if(s<=1){clearInterval(countdownTimer.current);return 0;}return s-1;});},1000);
      };
      const confirmFourEyes=()=>{
        setFourEyesPending(false);
        if(countdownTimer.current)clearInterval(countdownTimer.current);
        setIncidents(prev=>prev.map(inc=>({...inc,awaitingConfirm:false})));
      };
      const toggleLink=()=>{
        if(linkStatus==='ONLINE'){
          setLinkStatus('DEGRADED');
          setQueuedEvents(0);
          const t=setInterval(()=>setQueuedEvents(s=>{if(s>=4){clearInterval(t);return s;}return s+1;}),1800);
        }else{
          setLinkStatus('ONLINE');
          setTimeout(()=>setQueuedEvents(0),2000);
        }
      };
      const pageTitles={'command-center':'Command Center','live-surveillance':'Live Surveillance','border-map':'Border Map','incidents':'Incidents','evidence':'Evidence','analytics':'Analytics','camera-health':'Camera Health','audit-trail':'Audit Trail','account':'Account'};
      return (
        <div className="dashboard-container">
          <div id="sidebar-trigger" className="sidebar-trigger"
            onMouseEnter={()=>{if(closeTimer.current){clearTimeout(closeTimer.current);closeTimer.current=null;}setSidebarOpen(true);}}>
          </div>
          <aside id="sidebar" className={`sidebar${sidebarOpen?' sidebar--open':''}`}
            onMouseEnter={()=>{if(closeTimer.current){clearTimeout(closeTimer.current);closeTimer.current=null;}}}
            onMouseLeave={()=>{closeTimer.current=setTimeout(()=>{setSidebarOpen(false);closeTimer.current=null;},200);}}>
            <div className="sidebar-header">
              <ShieldIcon size={22} color="#78A578"/>
              <div><h1>SEEMA-DRISHTI</h1><div className="sidebar-sub">Border Intelligence</div></div>
            </div>
            <nav className="sidebar-nav">
              {sidebarItems.map(item=>(
                <button key={item.id} className={`sidebar-nav-item${activeView===item.id?' sidebar-nav-item--active':''}`}
                  onClick={()=>{setActiveView(item.id);setSidebarOpen(false);}}>
                  <span>{item.label}</span>
                  {item.badge!=null&&<span className="sidebar-badge">{item.badge}</span>}
                </button>
              ))}
            </nav>
            <div className="sidebar-footer">
              <div className="sidebar-footer-status">
                <div className={`status-dot status-dot--sm${linkStatus!=='ONLINE'?' status-dot--warn':''}`}></div>
                {linkStatus==='ONLINE'?'All systems online':'Link degraded'}
              </div>
              <div className="sidebar-footer-op" style={{cursor:'pointer',display:'flex',alignItems:'center',gap:'6px'}} onClick={()=>{setActiveView('account');setSidebarOpen(false);}}><svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="8" r="4"/><path d="M4 20c0-4 3.6-7 8-7s8 3 8 7"/></svg>Operator: BOP-07</div>
            </div>
          </aside>
          <main className="main-content">
            <header className="top-bar">
              <div style={{display:'flex',alignItems:'center',gap:'16px'}}>
                <button className="menu-toggle" onClick={()=>setSidebarOpen(s=>!s)}><MenuIcon/></button>
                <h2 className="top-bar-title">{pageTitles[activeView]||activeView}</h2>
              </div>
              <div className="top-bar-right">
                <FlipCard isFlipped={!isDark} onFlip={toggleMode}/>
                {fourEyesPending&&(<div className="four-eyes-topbar" onClick={()=>setActiveView('command-center')}>4-EYES REQ / {fourEyesCountdown}s</div>)}
                {queuedEvents>0&&(<div className="queued-topbar">{queuedEvents} event{queuedEvents>1?'s':''} queued</div>)}
                <div className="top-bar-meta">
                  <span>Sector 07</span><span className="meta-sep">/</span>
                  <span>{dateStr}</span><span className="meta-sep">/</span>
                  <span className={`meta-online${linkStatus!=='ONLINE'?' meta-degraded':''}`}>{linkStatus}</span>
                </div>
              </div>
            </header>
            {activeView==='command-center'&&(
              <>
                <div className="stats-row">
                  {stats.map((s,i)=>(
                    <div key={i} className="stat-card">
                      <div className="stat-card-label">{s.label}</div>
                      <div className="stat-card-val" style={s.label==='INCIDENTS'&&incidentSimulated?{color:'#ff4a4a'}:{}}>{s.val}</div>
                      <div className="stat-card-sub" style={s.subColor?{color:s.subColor}:{}}>{s.sub}</div>
                    </div>
                  ))}
                </div>
                <div className="dash-grid">
                  <div className="panel border-map-panel">
                    <div className="panel-head"><span className="panel-head-title">BORDER MAP</span><span className="panel-head-right">Sector 07</span></div>
                    <div className="border-map-area">
                      <div className="map-grid"></div>
                      <svg className="map-fence" viewBox="0 0 100 100" preserveAspectRatio="none">
                        <line x1="20" y1="80" x2="80" y2="30" stroke="rgba(120,165,120,0.15)" strokeWidth="0.4" strokeDasharray="2 2"/>
                      </svg>
                      {cameras.map(c=>(
                        <div key={c.id} className={`map-cam map-cam--${c.status}`} style={{left:`${c.x}%`,top:`${c.y}%`}}>
                          <div className="map-cam-dot"></div>
                          <span className="map-cam-label">{c.id}{c.status==='incident'?' / INCIDENT':''}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                  <div className="panel incidents-panel">
                    <div className="panel-head">
                      <span className="panel-head-title">PRIORITY INCIDENTS</span>
                      <button className="link-toggle-btn" onClick={toggleLink}>{linkStatus==='ONLINE'?'SIM LINK DROP':'RESTORE LINK'}</button>
                    </div>
                    <div className="incident-list">
                      {incidents.map((inc,i)=>(
                        <div key={inc.id} className={`incident-row${inc.awaitingConfirm?' incident-row--pending':''}`}>
                          <div className="incident-sev" style={{background:severityColor[inc.severity]}}></div>
                          <div className="incident-info">
                            <div className="incident-title">{inc.id} - {inc.desc}</div>
                            <div className="incident-meta">{inc.cameras} / {inc.time}</div>
                            <div className="incident-tags">
                              <span className="intent-tag" style={{borderColor:intentColor[inc.intent]||'#888',color:intentColor[inc.intent]||'#888'}}>{inc.intent}</span>
                              {queuedEvents>0&&i===0&&<span className="queued-tag">QUEUED</span>}
                              {inc.cam07t&&<span className="reid-tag">NON-FRT TRACKED</span>}
                            </div>
                          </div>
                          <div className="incident-right">
                            <div className="incident-scores">
                              <div>AI <span className="score-val">{inc.ai}%</span></div>
                              <div>EV <span className="score-val">{inc.ev}%</span></div>
                            </div>
                            {inc.awaitingConfirm&&fourEyesPending&&(<button className="confirm-btn" onClick={confirmFourEyes}>CONFIRM</button>)}
                          </div>
                        </div>
                      ))}
                    </div>
                    {fourEyesPending&&(
                      <div className="four-eyes-bar">
                        <div className="four-eyes-icon">2</div>
                        <div><div className="four-eyes-title">SECOND AUTHORIZATION REQUIRED</div><div className="four-eyes-sub">SLA expires in {fourEyesCountdown}s</div></div>
                        <button className="confirm-btn" onClick={confirmFourEyes}>AUTHORIZE</button>
                      </div>
                    )}
                  </div>
                </div>
              </>
            )}
            {activeView==='live-surveillance'&&(
              <div className="live-surveillance-grid">
                <div className={`cam-tile${incidentSimulated?' cam-tile--incident':''}`}>
                  <div className="cam-tile-header">
                    <span>CAM-07 / SECTOR 7 ALPHA</span>
                    <div style={{display:'flex',gap:'8px',alignItems:'center'}}>
                      {camActive&&tamperStatus!=='NOMINAL'&&<span className="tamper-badge">{tamperStatus}</span>}
                      {camActive&&freezeDetected&&<span className="freeze-badge">FREEZE/REPLAY</span>}
                      {camActive&&<span className="cam-live-dot">LIVE</span>}
                    </div>
                  </div>
                  <div className="cam-tile-body">
                    {!camActive?(
                      <div className="cam-placeholder" onClick={startCamera}>
                        <CameraIcon size={32} color="rgba(255,255,255,0.3)"/>
                        <span>TAP TO ACTIVATE FEED</span>
                        <span style={{fontSize:'11px',opacity:0.5,marginTop:'4px'}}>Tamper + Freeze detection active on live feed</span>
                      </div>
                    ):(
                      <>
                        <video ref={videoRef} autoPlay playsInline muted className="cam-video"></video>
                        <div className="cam-overlay">
                          <div className="cam-scanline"></div>
                          <div className="cam-crosshair"></div>
                          {tamperStatus!=='NOMINAL'&&<div className="cam-tamper-banner">{tamperStatus}</div>}
                          {freezeDetected&&<div className="cam-freeze-banner">FEED FROZEN / REPLAY SUSPECTED</div>}
                          {incidentSimulated&&<div className="cam-alert-overlay">THREAT DETECTED</div>}
                          {!incidentSimulated&&<button className="simulate-btn" onClick={simulateIncident}>SIMULATE AI TRIGGER</button>}
                        </div>
                      </>
                    )}
                  </div>
                  {camActive&&(
                    <div className="cam-integrity-row">
                      <span className={`integrity-chip${tamperStatus==='NOMINAL'?' integrity-chip--ok':' integrity-chip--warn'}`}>TAMPER: {tamperStatus}</span>
                      <span className={`integrity-chip${freezeDetected?' integrity-chip--warn':' integrity-chip--ok'}`}>FREEZE: {freezeDetected?'DETECTED':'CLEAR'}</span>
                      <span className="integrity-chip integrity-chip--info">NON-FRT RE-ID: ACTIVE</span>
                    </div>
                  )}
                </div>
                <div className="cam-tile">
                  <div className="cam-tile-header"><span>CAM-08 / INCIDENT ZONE</span><span className="cam-live-dot" style={{background:'#ff4a4a'}}>SIGNAL LOST</span></div>
                  <div className="cam-tile-body">
                    <div className="cam-placeholder">
                      <div style={{fontSize:'13px',color:'#ff4a4a',fontWeight:600,letterSpacing:'1px'}}>TAMPER / BLACKOUT DETECTED</div>
                      <div style={{fontSize:'11px',color:'rgba(255,74,74,0.6)',marginTop:'6px'}}>Last frame: 14:31:04</div>
                      <div style={{fontSize:'10px',color:'rgba(255,74,74,0.4)',marginTop:'4px'}}>Integrity score: FAILED</div>
                    </div>
                  </div>
                </div>
                <div className="cam-tile">
                  <div className="cam-tile-header"><span>CAM-09 / PERIMETER B</span><span className="cam-live-dot">LIVE</span></div>
                  <div className="cam-tile-body">
                    <div className="cam-placeholder">
                      <div style={{fontSize:'11px',opacity:0.4}}>ENCRYPTED FEED</div>
                      <div style={{fontSize:'10px',opacity:0.3,marginTop:'4px'}}>Signal: 91% / Tamper: NOMINAL</div>
                    </div>
                  </div>
                </div>
              </div>
            )}
            {activeView==='camera-health'&&(
              <div className="camera-health-grid">
                {camHealthData.map(cam=>(
                  <div key={cam.id} className={`cam-health-card cam-health-card--${cam.status}`}>
                    <div className="cam-health-card-header">
                      <span className="cam-health-id">{cam.id}</span>
                      <span className={`cam-health-status-dot cam-health-status-dot--${cam.status}`}></span>
                    </div>
                    <div className="cam-health-metrics">
                      <div className="cam-health-metric">
                        <span className="cam-health-metric-label">SIGNAL</span>
                        <div className="cam-health-bar-bg"><div className="cam-health-bar-fill" style={{width:`${cam.signal}%`,background:cam.signal>80?'#78A578':cam.signal>40?'#e8a838':'#ff4a4a'}}></div></div>
                        <span className="cam-health-metric-val">{cam.signal}%</span>
                      </div>
                      <div className="cam-health-metric">
                        <span className="cam-health-metric-label">TAMPER STATUS</span>
                        <span className={`cam-health-badge${cam.tamper==='NOMINAL'?' cam-health-badge--ok':' cam-health-badge--warn'}`}>{cam.tamper}</span>
                      </div>
                      <div className="cam-health-metric">
                        <span className="cam-health-metric-label">FREEZE DETECT</span>
                        <span className={`cam-health-badge${cam.freeze==='CLEAR'?' cam-health-badge--ok':' cam-health-badge--warn'}`}>{cam.freeze}</span>
                      </div>
                      <div className="cam-health-metric">
                        <span className="cam-health-metric-label">LAST VERIFIED</span>
                        <span className="cam-health-metric-val cam-health-metric-val--mono">{cam.lastVerified}</span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
            {activeView==='analytics'&&(
              <div className="analytics-grid">
                <div className="panel analytics-panel">
                  <div className="panel-head"><span className="panel-head-title">FALSE POSITIVE ACCOUNTABILITY</span><span className="panel-head-right">LAST 30 DAYS</span></div>
                  <div className="fp-list">
                    {fpStats.map(fp=>(
                      <div key={fp.cam} className="fp-row">
                        <div className="fp-cam-label">{fp.cam}</div>
                        <div className="fp-breakdown">
                          {fp.breakdown.map((b,i)=>(
                            <div key={i} className="fp-item">
                              <div className="fp-bar-bg"><div className="fp-bar-fill" style={{width:`${b.pct}%`}}></div></div>
                              <span className="fp-label">{b.label}</span>
                              <span className="fp-pct">{b.pct}%</span>
                            </div>
                          ))}
                        </div>
                        <div className="fp-summary">{fp.fp}/{fp.total} alerts were false positives ({fp.pct}%)</div>
                      </div>
                    ))}
                  </div>
                </div>
                <div className="panel analytics-panel">
                  <div className="panel-head"><span className="panel-head-title">INTENT CLASSIFICATION</span><span className="panel-head-right">LOITERING vs CROSSING</span></div>
                  <div className="intent-table">
                    <div className="intent-table-header"><span>ID</span><span>CLASSIFICATION</span><span>CAMERAS</span><span>DURATION</span></div>
                    {[
                      {id:'#1047',intent:'CROSSING',cameras:'CAM-07>08>09',duration:'3m 12s'},
                      {id:'#1046',intent:'LOITERING',cameras:'CAM-12',duration:'18m 40s'},
                      {id:'#1044',intent:'RECON-PATTERN',cameras:'CAM-07,09',duration:'47m'},
                      {id:'#1043',intent:'CROSSING',cameras:'CAM-03>07',duration:'1m 55s'},
                      {id:'#1042',intent:'STATIC',cameras:'CAM-09',duration:'22m'},
                    ].map(r=>(
                      <div key={r.id} className="intent-table-row">
                        <span className="intent-id">{r.id}</span>
                        <span className="intent-tag intent-tag--table" style={{borderColor:intentColor[r.intent]||'#888',color:intentColor[r.intent]||'#888'}}>{r.intent}</span>
                        <span className="intent-cams">{r.cameras}</span>
                        <span className="intent-dur">{r.duration}</span>
                      </div>
                    ))}
                  </div>
                </div>
                <div className="panel analytics-panel analytics-panel--full">
                  <div className="panel-head"><span className="panel-head-title">NON-BIOMETRIC RE-IDENTIFICATION</span><span className="reid-privacy-badge">NO FACIAL RECOGNITION</span></div>
                  <div className="reid-list">
                    {reidTracks.map(t=>(
                      <div key={t.id} className="reid-row">
                        <div className="reid-subject-id" style={{color:t.color}}>[{t.id}]</div>
                        <div className="reid-path">{t.path.map((p,i)=>(<span key={i}>{p}{i<t.path.length-1&&<span className="reid-arrow"> >> </span>}</span>))}</div>
                        <div className="reid-sig">Signature: {t.signature}</div>
                        <div className="reid-note">{t.note}</div>
                      </div>
                    ))}
                    <div className="reid-disclaimer">Subjects are tracked using gait analysis, motion vectors, clothing color profile, and silhouette height/stride. No biometric facial data is captured, stored, or processed. Compliant with applicable privacy directives.</div>
                  </div>
                </div>
              </div>
            )}

            {activeView==='border-map'&&(
              <div className="border-map-full">
                <div className="bm-map-area" style={{position:'relative'}}>
                  <div ref={mapRef} style={{width:'100%',height:'100%',minHeight:'420px',borderRadius:'10px',zIndex:1}}></div>
                  <div className="bm-map-overlay-btns">
                    <button className="bm-locate-btn" onClick={handleLocate}>
                      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="3"/><path d="M12 2v3M12 19v3M2 12h3M19 12h3"/></svg>
                      MY LOCATION
                    </button>
                    {userCoords&&<div className="bm-user-coords">{userCoords}</div>}
                  </div>
                </div>
                <div className="bm-sidebar">
                  <div className="bm-sidebar-section">
                    <div className="bm-sidebar-title">ACTIVE NODES</div>
                    {[
                      {id:'CAM-07',status:'ONLINE',signal:98,lat:'28.263N',lon:'72.553E'},
                      {id:'CAM-08',status:'INCIDENT',signal:0,lat:'28.242N',lon:'72.568E'},
                      {id:'CAM-09',status:'ONLINE',signal:91,lat:'28.273N',lon:'72.588E'},
                      {id:'CAM-12',status:'DEGRADED',signal:73,lat:'28.232N',lon:'72.544E'},
                      {id:'CAM-03',status:'ONLINE',signal:87,lat:'28.284N',lon:'72.608E'},
                      {id:'BOP-07',status:'ONLINE',signal:99,lat:'28.236N',lon:'72.598E'},
                    ].map(n=>(
                      <div key={n.id} className="bm-node-row">
                        <div className={`bm-node-status bm-node-status--${n.status.toLowerCase()}`}>{n.id}</div>
                        <div className="bm-node-detail">
                          <div className="bm-node-coords">{n.lat} {n.lon}</div>
                          <div className="bm-node-signal">Signal: {n.signal}%</div>
                        </div>
                      </div>
                    ))}
                  </div>
                  <div className="bm-sidebar-section">
                    <div className="bm-sidebar-title">SECTOR INFO</div>
                    <div className="bm-info-row"><span>Zone</span><span>RESTRICTED-A</span></div>
                    <div className="bm-info-row"><span>Sector</span><span>07-ALPHA</span></div>
                    <div className="bm-info-row"><span>Fence Length</span><span>14.3 km</span></div>
                    <div className="bm-info-row"><span>Patrols Active</span><span>3</span></div>
                    <div className="bm-info-row"><span>Last Sweep</span><span>14:18 ZULU</span></div>
                  </div>
                </div>
              </div>
            )}

            {activeView==='incidents'&&(
              <div className="incidents-full-view">
                <div className="incidents-toolbar">
                  <button className="dl-report-btn" onClick={()=>{
                    const report=[
                      'SEEMA-DRISHTI INCIDENT REPORT',
                      'Sector: 07-Alpha | Generated: '+new Date().toISOString(),
                      '---',
                      '#1047 | CROSSING | CAM-07>08>09 | 14:31 | AI:91% | HIGH',
                      '#1046 | LOITERING | CAM-12 | 14:28 | AI:75% | MEDIUM',
                      '#1045 | STATIC | CAM-03 | 13:52 | AI:64% | LOW',
                      '#1044 | RECON-PATTERN | CAM-07,09 | 13:30 | AI:88% | HIGH',
                      '#1043 | STATIC | CAM-01 | 12:44 | AI:74% | MEDIUM',
                      '#1042 | CROSSING | CAM-09 | 02:31 | AI:61% | LOW',
                    ].join('\n');
                    const blob=new Blob([report],{type:'text/plain'});
                    const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='incident-report-sector07-'+Date.now()+'.txt';a.click();
                  }}>
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7,10 12,15 17,10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
                    DOWNLOAD REPORT
                  </button>
                  <div style={{display:'flex',gap:'8px'}}>
                    {['ALL','HIGH','MEDIUM','LOW'].map(f=>(
                      <button key={f} className="filter-btn filter-btn--active">{f}</button>
                    ))}
                  </div>
                  <div className="incidents-count">{incidents.length} incidents</div>
                </div>
                <div className="incidents-table-header">
                  <span>ID</span><span>DESCRIPTION</span><span>CLASSIFICATION</span>
                  <span>CAMERAS</span><span>TIME</span><span>AI</span><span>EV</span><span>STATUS</span>
                </div>
                <div className="incidents-table-body">
                  {incidents.map((inc,i)=>(
                    <div key={inc.id} className={`incidents-table-row incidents-table-row--${inc.severity}`}>
                      <span className="inc-id">{inc.id}</span>
                      <span className="inc-desc">{inc.desc}</span>
                      <span><span className="intent-tag" style={{borderColor:({CROSSING:'#ff4a4a',LOITERING:'#e8a838',STATIC:'#5890c8','RECON-PATTERN':'#c85890'})[inc.intent]||'#888',color:({CROSSING:'#ff4a4a',LOITERING:'#e8a838',STATIC:'#5890c8','RECON-PATTERN':'#c85890'})[inc.intent]||'#888'}}>{inc.intent}</span></span>
                      <span className="inc-cams">{inc.cameras}</span>
                      <span className="inc-time">{inc.time}</span>
                      <span className="inc-score">{inc.ai}%</span>
                      <span className="inc-score">{inc.ev}%</span>
                      <span><span className={`inc-status-badge inc-status-badge--${inc.awaitingConfirm?'pending':'open'}`}>{inc.awaitingConfirm?'PENDING AUTH':'OPEN'}</span></span>
                    </div>
                  ))}
                  {[
                    {id:'#1044',desc:'Recon pattern - north fence',cameras:'CAM-07, CAM-09',time:'13:30:22',ai:88,ev:81,severity:'high',intent:'RECON-PATTERN',awaitingConfirm:false},
                    {id:'#1043',desc:'Unauthorized vehicle near gate',cameras:'CAM-03',time:'12:44:18',ai:74,ev:69,severity:'medium',intent:'STATIC',awaitingConfirm:false},
                    {id:'#1042',desc:'Night movement thermal signature',cameras:'CAM-09',time:'02:11:55',ai:61,ev:53,severity:'low',intent:'CROSSING',awaitingConfirm:false},
                  ].map(inc=>(
                    <div key={inc.id} className={`incidents-table-row incidents-table-row--${inc.severity}`}>
                      <span className="inc-id">{inc.id}</span>
                      <span className="inc-desc">{inc.desc}</span>
                      <span><span className="intent-tag" style={{borderColor:({CROSSING:'#ff4a4a',LOITERING:'#e8a838',STATIC:'#5890c8','RECON-PATTERN':'#c85890'})[inc.intent]||'#888',color:({CROSSING:'#ff4a4a',LOITERING:'#e8a838',STATIC:'#5890c8','RECON-PATTERN':'#c85890'})[inc.intent]||'#888'}}>{inc.intent}</span></span>
                      <span className="inc-cams">{inc.cameras}</span>
                      <span className="inc-time">{inc.time}</span>
                      <span className="inc-score">{inc.ai}%</span>
                      <span className="inc-score">{inc.ev}%</span>
                      <span><span className="inc-status-badge inc-status-badge--open">OPEN</span></span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {activeView==='evidence'&&(
              <div className="evidence-grid">
                {[
                  {inc:'#1047',type:'FRAME CAPTURE',cam:'CAM-07',time:'14:31:04',size:'2.4 MB',res:'1920x1080',custody:['14:31:05 System auto-capture','14:32:10 Operator BOP-07 reviewed','14:35:00 Flagged for escalation']},
                  {inc:'#1047',type:'VIDEO CLIP',cam:'CAM-08',time:'14:31:00',size:'18.7 MB',res:'1280x720 / 30s',custody:['14:31:00 System auto-record','14:32:45 Operator BOP-07 verified']},
                  {inc:'#1047',type:'METADATA LOG',cam:'CAM-07',time:'14:31:04',size:'12 KB',res:'JSON / AI scores',custody:['14:31:04 System generated','14:33:00 Integrity hash verified']},
                  {inc:'#1046',type:'FRAME CAPTURE',cam:'CAM-12',time:'14:28:04',size:'2.1 MB',res:'1920x1080',custody:['14:28:05 System auto-capture','14:29:30 Operator BOP-07 reviewed']},
                  {inc:'#1044',type:'VIDEO CLIP',cam:'CAM-07',time:'13:30:22',size:'42.1 MB',res:'1280x720 / 80s',custody:['13:30:22 System auto-record','13:31:55 Flagged by AI pattern engine']},
                  {inc:'#1043',type:'FRAME CAPTURE',cam:'CAM-03',time:'12:44:18',size:'1.9 MB',res:'1920x1080',custody:['12:44:19 System auto-capture']},
                ].map((ev,i)=>(
                  <div key={i} className="evidence-card">
                    <div className="evidence-card-header">
                      <span className="evidence-type">{ev.type}</span>
                      <span className="evidence-inc-ref">{ev.inc}</span>
                    </div>
                    <div className="evidence-card-body">
                      <div className="evidence-field"><span>Camera</span><span>{ev.cam}</span></div>
                      <div className="evidence-field"><span>Timestamp</span><span>{ev.time}</span></div>
                      <div className="evidence-field"><span>Size</span><span>{ev.size}</span></div>
                      <div className="evidence-field"><span>Details</span><span>{ev.res}</span></div>
                    </div>
                    <div className="evidence-custody">
                      <div className="evidence-custody-title">CHAIN OF CUSTODY</div>
                      {ev.custody.map((c,j)=>(
                        <div key={j} className="evidence-custody-entry">{c}</div>
                      ))}
                    </div>
                    <button className="ev-download-btn" onClick={()=>{
                      const data=JSON.stringify({incident:ev.inc,type:ev.type,camera:ev.cam,timestamp:ev.time,size:ev.size,resolution:ev.res,chain_of_custody:ev.custody},null,2);
                      const blob=new Blob([data],{type:'application/json'});
                      const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='evidence-'+ev.inc.replace('#','')+'-'+ev.type.toLowerCase().replace(' ','-')+'.json';a.click();
                    }}>
                      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7,10 12,15 17,10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
                      DOWNLOAD
                    </button>
                  </div>
                ))}
              </div>
            )}

            {activeView==='audit-trail'&&(
              <div className="audit-trail-view">
                <div className="audit-toolbar">
                  <div className="audit-title-info">System event log / All operators / Sector 07</div>
                  <div className="audit-count">24 events today</div>
                </div>
                <div className="audit-log">
                  {[
                    {time:'14:35:00',type:'ESCALATION',actor:'BOP-07',action:'Incident #1047 flagged for escalation. Second authorization pending.'},
                    {time:'14:33:10',type:'AI-ALERT',actor:'SYSTEM',action:'Intent reclassified: CAM-07 subject pattern changed from STATIC to CROSSING.'},
                    {time:'14:31:04',type:'INCIDENT',actor:'SYSTEM',action:'Incident #1047 created. AI score 91%, EV score 86%. CAM-07>CAM-08>CAM-09.'},
                    {time:'14:31:00',type:'TAMPER',actor:'SYSTEM',action:'CAM-08 signal lost. Tamper/blackout detection triggered. Integrity score: FAILED.'},
                    {time:'14:28:04',type:'INCIDENT',actor:'SYSTEM',action:'Incident #1046 created. Loitering detected at CAM-12. 9 direction reversals in 18m.'},
                    {time:'14:20:00',type:'AUTH',actor:'BOP-07',action:'Operator BOP-07 authenticated. Session token issued.'},
                    {time:'14:15:33',type:'SYSTEM',actor:'SYSTEM',action:'All cameras nominal. Perimeter integrity check passed.'},
                    {time:'14:02:11',type:'SYSTEM',actor:'SYSTEM',action:'CAM-09 baseline reference frame updated. Tamper detection reset.'},
                    {time:'14:00:00',type:'SYSTEM',actor:'SYSTEM',action:'Scheduled integrity verification completed. 5/6 cameras NOMINAL. 1 DEGRADED (CAM-12).'},
                    {time:'13:52:11',type:'INCIDENT',actor:'SYSTEM',action:'Incident #1045 created. Unattended object detected at CAM-03. AI 64%, EV 58%.'},
                    {time:'13:30:22',type:'INCIDENT',actor:'SYSTEM',action:'Incident #1044 created. Recon pattern on north fence. 5 perimeter approach vectors in 47m.'},
                    {time:'13:00:00',type:'SYSTEM',actor:'SYSTEM',action:'Edge sync completed. 0 queued events pending.'},
                    {time:'12:44:18',type:'INCIDENT',actor:'SYSTEM',action:'Incident #1043 created. Unauthorized vehicle detected at CAM-03. AI 74%, EV 69%.'},
                    {time:'06:00:00',type:'SYSTEM',actor:'SYSTEM',action:'System boot. All 6 camera nodes connected. Non-FRT re-ID engine initialized.'},
                  ].map((entry,i)=>(
                    <div key={i} className="audit-entry">
                      <div className="audit-entry-time">{entry.time}</div>
                      <div className={`audit-entry-type audit-entry-type--${entry.type.toLowerCase().replace('-','')}`}>{entry.type}</div>
                      <div className="audit-entry-actor">{entry.actor}</div>
                      <div className="audit-entry-action">{entry.action}</div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {activeView==='account'&&(
              <div className="account-view">
                <div className="account-profile-card">
                  <div className="account-avatar">B7</div>
                  <div className="account-info">
                    <div className="account-name">OPERATOR BOP-07</div>
                    <div className="account-rank">Sub-Inspector / Sector Commander</div>
                    <div className="account-badges">
                      <span className="account-badge account-badge--active">ACTIVE SESSION</span>
                      <span className="account-badge account-badge--clearance">CLEARANCE LVL 3</span>
                    </div>
                  </div>
                </div>
                <div className="account-grid">
                  <div className="account-section">
                    <div className="account-section-title">SESSION INFO</div>
                    <div className="account-field"><span>Login Time</span><span>14:00:00 ZULU</span></div>
                    <div className="account-field"><span>Shift</span><span>Alpha Shift (0800-2000)</span></div>
                    <div className="account-field"><span>Station</span><span>BOP-07 / Sector 07-Alpha</span></div>
                    <div className="account-field"><span>IP Terminal</span><span>10.7.0.14</span></div>
                    <div className="account-field"><span>Auth Method</span><span>RFID + Biometric</span></div>
                  </div>
                  <div className="account-section">
                    <div className="account-section-title">ACCESS PERMISSIONS</div>
                    <div className="account-field"><span>Live Surveillance</span><span className="perm-ok">GRANTED</span></div>
                    <div className="account-field"><span>Evidence Vault</span><span className="perm-ok">GRANTED</span></div>
                    <div className="account-field"><span>Incident Close</span><span className="perm-ok">GRANTED</span></div>
                    <div className="account-field"><span>4-Eyes Escalation</span><span className="perm-ok">GRANTED</span></div>
                    <div className="account-field"><span>System Config</span><span className="perm-deny">DENIED</span></div>
                    <div className="account-field"><span>Export Classified</span><span className="perm-deny">DENIED</span></div>
                  </div>
                  <div className="account-section">
                    <div className="account-section-title">ACTIVITY TODAY</div>
                    <div className="account-field"><span>Incidents Reviewed</span><span>6</span></div>
                    <div className="account-field"><span>Alerts Actioned</span><span>3</span></div>
                    <div className="account-field"><span>Evidence Downloads</span><span>2</span></div>
                    <div className="account-field"><span>4-Eyes Escalations</span><span>1</span></div>
                    <div className="account-field"><span>Camera Reviews</span><span>12</span></div>
                  </div>
                  <div className="account-section">
                    <div className="account-section-title">DOWNLOAD FULL REPORT</div>
                    <div style={{padding:'16px 0',display:'flex',flexDirection:'column',gap:'10px'}}>
                      <button className="account-dl-btn" onClick={()=>{
                        const r='SEEMA-DRISHTI OPERATOR REPORT\nOperator: BOP-07\nDate: '+new Date().toDateString()+'\nShift: Alpha 0800-2000\nIncidents: 6 reviewed\nEscalations: 1\nStation: BOP-07 Sector 07-Alpha';
                        const b=new Blob([r],{type:'text/plain'});const a=document.createElement('a');a.href=URL.createObjectURL(b);a.download='operator-report-BOP07-'+Date.now()+'.txt';a.click();
                      }}>DOWNLOAD SHIFT REPORT</button>
                      <button className="account-dl-btn account-dl-btn--secondary" onClick={()=>{
                        const r=JSON.stringify({operator:'BOP-07',date:new Date().toISOString(),incidents_reviewed:6,alerts_actioned:3,escalations:1,evidence_downloads:2,camera_reviews:12},null,2);
                        const b=new Blob([r],{type:'application/json'});const a=document.createElement('a');a.href=URL.createObjectURL(b);a.download='activity-log-BOP07-'+Date.now()+'.json';a.click();
                      }}>DOWNLOAD ACTIVITY LOG (JSON)</button>
                    </div>
                  </div>
                </div>
              </div>
            )}

          </main>
        </div>
      );
    }



    function App(){
      const [appState,setAppState]=useState('landing');
      const [prevState,setPrevState]=useState('landing');
      const [time,setTime]=useState(new Date().toLocaleTimeString('en-US',{hour12:false})+" ZULU");
      useEffect(()=>{
        const t=setInterval(()=>setTime(new Date().toLocaleTimeString('en-US',{hour12:false})+" ZULU"),1000);
        return()=>clearInterval(t);
      },[]);
      useEffect(()=>{
        if(appState==='dashboard'){if(window.setAuroraState)window.setAuroraState(0);if(window.showGridBg)window.showGridBg(true);}
        else{if(window.showGridBg)window.showGridBg(false);}
      },[appState]);
      const goLegal=(page)=>{ setPrevState(appState); setAppState(page); };
      const goBack=()=>setAppState(prevState);
      if(appState==='privacy')return<PrivacyPage onBack={goBack}/>;
      if(appState==='terms')return<TermsPage onBack={goBack}/>;
      if(appState==='landing')return<LandingPage onEnter={()=>setAppState('signin')} onPrivacy={()=>goLegal('privacy')} onTerms={()=>goLegal('terms')}/>;
      if(appState==='signin')return<SignInPage onSignIn={()=>setAppState('dashboard')}/>;
      return<Dashboard time={time}/>;    }

    ReactDOM.createRoot(document.getElementById('root')).render(<App/>);
