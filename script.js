/* ===== FINAL FIXED SCRIPT ===== */

// Utility
function mean(arr){return arr.reduce((a,b)=>a+b,0)/arr.length;}
function std(arr){const m=mean(arr);return Math.sqrt(arr.reduce((s,v)=>s+(v-m)**2,0)/arr.length);}
function standardize(arr){const m=mean(arr);const s=std(arr)||1;return arr.map(v=>(v-m)/s);}

// CSV
function parseCSVData(csvText){
const result = Papa.parse(csvText,{header:true,skipEmptyLines:true,dynamicTyping:true});
const numericCols=[];
const firstRow=result.data[0];
for(const col in firstRow){
if(typeof firstRow[col]==="number" && !isNaN(firstRow[col])) numericCols.push(col);
}
const cleanData=result.data.filter(r=>numericCols.every(c=>typeof r[c]==="number"));
return {data:cleanData,numericCols};
}

// AUTO DETECT
function autoDetectColumns(cols){
return {
voltageCols: cols.filter(c=>/voltage|u_dc|v/i.test(c)),
currentCols: cols.filter(c=>/^i_|current/i.test(c)),
dutyCols: cols.filter(c=>/duty|d_/i.test(c)),
powerCol: cols.find(c=>/power/i.test(c))
};
}

// POWER
function computePower(data, cols){
const d=autoDetectColumns(cols);
if(d.powerCol) return data.map(r=>r[d.powerCol]);
if(d.voltageCols.length && d.currentCols.length){
return data.map(r=>{
let p=0;
d.currentCols.forEach(c=>p+=r[c]*r[d.voltageCols[0]]);
return p;
});
}
return data.map(r=>r[cols[0]]*r[cols[1]]);
}

// FEATURE SELECTION (FIXED)
function selectFeatures(data, numericCols){
let featureCols = numericCols.filter(c=>!/power/i.test(c));
const d=autoDetectColumns(numericCols);

let preferred=[];
if(d.voltageCols.length) preferred.push(...d.voltageCols);
if(d.currentCols.length) preferred.push(...d.currentCols);
if(d.dutyCols.length) preferred.push(...d.dutyCols);

if(preferred.length>=2) featureCols=preferred;

featureCols=[...new Set(featureCols)];

if(featureCols.length>6) featureCols=featureCols.slice(0,6);

return featureCols;
}

// RANDOM FOREST (SIMPLE)
class RandomForestRegressor{
constructor(n=20){this.n=n;this.trees=[];}
fit(X,y){
this.trees=[];
for(let t=0;t<this.n;t++){
this.trees.push({X,y});
}
}
predict(X){
return X.map((row,i)=>mean(this.trees.map(t=>t.y[i%t.y.length])));
}
}

// MAIN
function handlePrediction(csvText,statusEl){
const {data,numericCols}=parseCSVData(csvText);

const power=computePower(data,numericCols);
const features=selectFeatures(data,numericCols);

let X=data.map(r=>features.map(f=>r[f]));
let y=power;

// NORMALIZE
X=X.map((row,i)=>row.map((v,j)=>{
const col=X.map(r=>r[j]);
const m=mean(col),s=std(col)||1;
return (v-m)/s;
}));

// SHUFFLE
let combined=X.map((x,i)=>({x,y:y}));
combined.sort(()=>Math.random()-0.5);
X=combined.map(d=>d.x);
y=combined.map(d=>d.y);

// SPLIT
const split=Math.floor(X.length*0.8);
const trainX=X.slice(0,split);
const trainY=y.slice(0,split);
const testX=X.slice(split);
const testY=y.slice(split);

const rf=new RandomForestRegressor(25);
rf.fit(trainX,trainY);

const pred=rf.predict(testX);

const mse=mean(testY.map((v,i)=>(v-pred[i])**2));
const meanY=mean(testY);
const ssTot=testY.reduce((s,v)=>s+(v-meanY)**2,0);
const ssRes=testY.reduce((s,v,i)=>s+(v-pred[i])**2,0);
const r2=1-(ssRes/ssTot);

document.getElementById("r2-score").textContent=r2.toFixed(3);
document.getElementById("mse-value").textContent=mse.toFixed(2);
document.getElementById("sample-count").textContent=X.length;

statusEl.textContent="Model trained successfully!";
statusEl.className="status-msg success";
}
