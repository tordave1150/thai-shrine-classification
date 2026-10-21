// Offline Markdown rendering and visual QA; no experiment execution.
const fs=require('fs'),path=require('path');
const {pathToFileURL}=require('url');
const {root,sharpRuntime}=require('./report_utils');
async function main(){
  const runtime=path.join(process.env.USERPROFILE,'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules');
  const {marked}=await import(pathToFileURL(require.resolve(path.join(runtime,'marked'))).href);
  const {chromium}=require(path.join(runtime,'playwright'));
  const markdown=fs.readFileSync(path.join(root,'README.md'),'utf8');
  const html='<!doctype html><html lang="th"><meta charset="utf-8"><base href="'+pathToFileURL(root+path.sep).href+'"><title>README preview</title><style>body{font:16px/1.7 Arial,Tahoma,sans-serif;color:#1f2937;background:white;margin:40px auto;max-width:1000px}h1,h2,h3,h4{line-height:1.35;margin-top:1.7em}h1{font-size:30px}h2{font-size:25px;border-bottom:1px solid #ddd;padding-bottom:10px}h3{font-size:21px}img{max-width:100%;height:auto}table{border-collapse:collapse;width:100%;margin:20px 0;font-size:14px}th,td{border:1px solid #d1d5db;padding:9px;overflow-wrap:anywhere}th{background:#f3f4f6;text-align:left}pre{background:#f3f4f6;padding:16px;white-space:pre-wrap;overflow-wrap:anywhere}code{font-size:.9em}a{color:#0066a6}details{padding:12px;border:1px solid #ddd}sub{font-size:14px}</style><body>'+marked.parse(markdown)+'</body></html>';
  const htmlFile=path.join(__dirname,'qa/readme_preview.html');fs.writeFileSync(htmlFile,html);
  const executable=fs.existsSync(chromium.executablePath())?chromium.executablePath():'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe';
  const browser=await chromium.launch({headless:true,executablePath:executable,args:['--allow-file-access-from-files']});
  try{
    const page=await browser.newPage({viewport:{width:1200,height:1000},deviceScaleFactor:1});
    await page.route(/^https?:/,route=>route.abort());
    await page.goto(pathToFileURL(htmlFile).href);
    await page.evaluate(async()=>{await document.fonts.ready;await Promise.all([...document.images].map(i=>i.decode().catch(()=>{})));});
    const result=await page.evaluate(()=>({images:[...document.images].length,broken_images:[...document.images].filter(i=>!i.complete||!i.naturalWidth).map(i=>i.src),horizontal_overflow:document.documentElement.scrollWidth>innerWidth}));
    if(result.broken_images.length||result.horizontal_overflow)throw Error(JSON.stringify(result));
    await page.screenshot({path:path.join(__dirname,'qa/readme_abstract.png')});
    await page.getByRole('heading',{name:'4.3 Selected configurations and final evaluation',exact:true}).scrollIntoViewIfNeeded();
    await page.screenshot({path:path.join(__dirname,'qa/readme_selected_configurations.png')});
    await page.getByRole('heading',{name:'6.2 Learning curves across five seeds',exact:true}).scrollIntoViewIfNeeded();
    await page.screenshot({path:path.join(__dirname,'qa/readme_learning_curves.png')});
    fs.writeFileSync(path.join(__dirname,'qa/readme_render_validation.json'),JSON.stringify({rendered_utc:new Date().toISOString(),...result,external_network_requests_blocked:true},null,2)+'\n');
    console.log('Offline README render passed: '+result.images+' images, no broken images or horizontal overflow.');
  }finally{await browser.close();}
  const sharp=sharpRuntime();
  for(const name of ['image_dimensions','brightness','colour_modes','learning_curves_resnet50'])await sharp(path.join(__dirname,'report_assets/latest',name+'.svg')).greyscale().png().toFile(path.join(__dirname,'qa',name+'_grayscale.png'));
  console.log('Rendered four grayscale proofs for manual review.');
}
main().catch(error=>{console.error(error);process.exitCode=1;});
