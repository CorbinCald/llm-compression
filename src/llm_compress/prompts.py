COMPRESSION_PROMPT = """You have received one block of code to compress into a small, recoverable component description.

Use this exact format for the component:
<-Name:name|Input:inputs or none|Return:exact behavior/code-shaped spec|Path:path|Order:order+indent->

Goal: use as few tokens as you can while preserving enough information for another LLM to rebuild equivalent code.

Rules:
- Keep exact names that other code may reference.
- Keep exact strings, numbers, selectors, globs, URLs, JSON shapes, and side effects.
- Prefer compact code-shaped specs in Return when prose would be ambiguous.
- Omit formatting and comments.
- If a component cannot be reconstructed from the compressed form, return ONLY the string, <FAILURE>.
- Return only ONE compressed component, no other verbiage.

Order field:
- Use file order plus indent letter: a = top-level, b = one indent, c = two indents, etc.
- Example: Order:4a means fourth top-level component in the file.

Examples of various compressed components:

<-Name:paintSquareClickHandlers|Input:outer squares,socket,color(live)|Return:squares.forEach((sq,i)=>sq.addEventListener('click',()=>socket.send(JSON.stringify({paint:{index:i,color}}))))|Path:./client/app.js|Order:12a->
<-Name:h1|Input:none|Return:h1{font-size:2.5rem;font-weight:700;letter-spacing:2px;margin:0 0 20px;background:linear-gradient(90deg,#ff6b6b,#feca57,#48dbfb,#1dd1a1);-webkit-background-clip:text;background-clip:text;color:transparent}|Path:./client/style.css|Order:2a->
<-Name:serverBootstrap|Input:none|Return:const express=require('express');const{WebSocketServer,WebSocket}=require('ws');const app=express();app.use(express.static('client'));const server=app.listen(3000,()=>console.log(`Server Running on 3000`));const wss=new WebSocketServer({server});const participants=[];const grid=[]|Path:./server.js|Order:1a->
<-Name:getPhotos|Input:count:int,size:num|Return:return Object.fromEntries(Array.from({length:count},(_,i)=>[`photo${i+1}`,`https://picsum.photos/${size}?random=${i+1}`]))|Path:./app/index.tsx|Order:4d->
<-Name:gitIgnore|Input:none|Return:ignore exactly:node_modules/ .expo/ dist/ web-build/ expo-env.d.ts .kotlin/ *.orig.* *.jks *.p8 *.p12 *.key *.mobileprovision .metro-health-check* npm-debug.* yarn-debug.* yarn-error.* .DS_Store *.pem .env*.local *.tsbuildinfo app-example /ios /android|Path:./.gitignore|Order:1a->
"""

DECOMPRESSION_PROMPT = """You rebuild source files from compressed component descriptions.

Input lines use this format:
<-Name:name|Input:inputs or none|Return:exact behavior/code-shaped spec|Path:path|Order:order+indent->

Goal: produce equivalent working code.

Rules:
- Treat Return as the source of truth.
- Preserve exact names, strings, numbers, selectors, globs, URLs, JSON shapes, and side effects.
- Fill in normal syntax, imports, braces, and formatting when needed.
- Do not add features that are not described.
- If two interpretations are possible, choose the simpler one that best matches the spec.
- If the spec is impossible or missing a required fact, return only <DECOMP_FAILURE>.
"""

COMPONENT_PROMPT_VERSION = "2026-05-16.v1"
