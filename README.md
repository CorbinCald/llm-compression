# LLM Compression

## Objective: For a given file or codebase, reduce chars at the cost of inference, while maintaining identical function.

### Component compression format

`<-Name:name|Input:inputs or none|Return:exact behavior/code-shaped spec|Path:path|Order:order+indent->`

`Order` is file order plus indent level, where `a` is top-level. Compression is only valid when expansion is recoverable: keep exact names, literals, selectors, globs, protocol shapes, formulas, and side effects. Prefer code-shaped specs over prose; omit only formatting/comments that do not affect behavior.

#### Examples of compressed components:

```
<-Name:paintSquareClickHandlers|Input:outer squares,socket,color(live)|Return:squares.forEach((sq,i)=>sq.addEventListener('click',()=>socket.send(JSON.stringify({paint:{index:i,color}}))))|Path:./client/app.js|Order:12a->

<-Name:h1|Input:none|Return:h1{font-size:2.5rem;font-weight:700;letter-spacing:2px;margin:0 0 20px;background:linear-gradient(90deg,#ff6b6b,#feca57,#48dbfb,#1dd1a1);-webkit-background-clip:text;background-clip:text;color:transparent}|Path:./client/style.css|Order:2a->

<-Name:serverBootstrap|Input:none|Return:const express=require('express');const{WebSocketServer,WebSocket}=require('ws');const app=express();app.use(express.static('client'));const server=app.listen(3000,()=>console.log(`Server Running on 3000`));const wss=new WebSocketServer({server});const participants=[];const grid=[]|Path:./server.js|Order:1a->

<-Name:getPhotos|Input:count:int,size:num|Return:return Object.fromEntries(Array.from({length:count},(_,i)=>[`photo${i+1}`,`https://picsum.photos/${size}?random=${i+1}`]))|Path:./app/index.tsx|Order:4d->

<-Name:gitIgnore|Input:none|Return:ignore exactly:node_modules/ .expo/ dist/ web-build/ expo-env.d.ts .kotlin/ *.orig.* *.jks *.p8 *.p12 *.key *.mobileprovision .metro-health-check* npm-debug.* yarn-debug.* yarn-error.* .DS_Store *.pem .env*.local *.tsbuildinfo app-example /ios /android|Path:./.gitignore|Order:1a->
```

## Transcoder: a transcoding function will take the ultraconcise description, pass it through a cost-effective LLM, and assemble it into the equivalent file. Function position and nesting will be handled programatically based on the the Path and Order artifacts.

## Tests: Unit tests ensure minimal lossiness upon compression and decompression.

## Iterability: The project will be improved iteratively inspired by **Karpathy's Autoresearch** (https://github.com/karpathy/autoresearch), with the loss metric being tests failed.

### To start, we can use ~7 public Github repos of sizes, compress them in parallel, decompress them, and measure the difference in tests passed or failed, after which point the LLM will iterate to improve test success. We are starting with 4 Small Repos, 2 Medium, and 1 Large repo.