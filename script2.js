var plotDiv = document.getElementsByClassName("plotly-graph-div js-plotly-plot")[0];

var selectioras = [];

document.querySelector("[id^='modebar'] > div:nth-child(4) > a:nth-child(3)").click()
document.querySelector("[id^='modebar'] > div:nth-child(4) > a:nth-child(2)").click()

plotDiv.data.filter(x => x.name == "Volume");

VolumePls = plotDiv.data.filter(x => x.name == "Volume")[0];

Enter_long = plotDiv.data.filter(x => x.name == "enter_long")[0];
Exit_long = plotDiv.data.filter(x => x.name == "exit_long")[0];
Enter_short = plotDiv.data.filter(x => x.name == "enter_short")[0];
Exit_short = plotDiv.data.filter(x => x.name == "exit_short")[0];

delete VolumePls.x;
delete VolumePls.y;
VolumePls.x = "";
VolumePls.y = "";
VolumePls.xaxis = "";
VolumePls.yaxis = "";

try {
	plotDiv._context.showTips = false
} catch (e) {}

//try {
	//plotDiv.data[0].increasing = {'fillcolor': '#50D217', 'line': {'color': '#50D217', 'width': '1.5'}}
	//plotDiv.data[0].decreasing = {'fillcolor': '#E22A29', 'line': {'color': '#E22A29', 'width': '1.5'}}
//}catch (e) {}

try {
	long = plotDiv.data.filter(x => x.name == "enter_long")[0];
	long.marker.color = '#009FFF'
	long.marker.size = 5
	long.marker.line.color = 'black'
	long.marker.line.width = 0.5
	long.marker.symbol = "circle"
	long.textposition = 'bottom center'
	long.mode = 'markers+text'
	long.textfont = {'family': 'tahoma', 'size': 18, 'color': '#009FFF'}
	long.texttemplate = '<br>🠙'
	//long.marker.line.symbol = "triangle-up"
}catch (e) {}

try {
	short = plotDiv.data.filter(x => x.name == "enter_short")[0];
	short.marker.color = '#FFD035'
	short.marker.size = 5
	short.marker.line.color = 'black'
	short.marker.line.width = 0.5
	short.marker.symbol = "circle"
	short.textposition = 'top center'
	short.mode = 'markers+text'
	short.textfont = {'family': 'tahoma', 'size': 18, 'color': '#FFD035'}
	short.texttemplate = '🠛<br>'
	//short.marker.line.symbol = "triangle-down"
}catch (e) {}

try {
	longexit = plotDiv.data.filter(x => x.name == "exit_long")[0];
	longexit.marker.color = 'rgba(255, 230, 0, 1)'
	longexit.marker.size = 9
	longexit.marker.line.width = 1
	longexit.marker.symbol = "x"
}catch (e) {}

try {
	shortexit = plotDiv.data.filter(x => x.name == "exit_short")[0];
	shortexit.marker.color = 'rgba(105, 160, 255, 1)'
	shortexit.marker.size = 9
	shortexit.marker.line.width = 1
	shortexit.marker.symbol = "x"
}catch (e) {}

try {
	exitprofit = plotDiv.data.filter(x => x.name == "Exit - Profit")[0];
	exitprofit.marker.color = 'rgba(0, 255, 0, 1)'
	exitprofit.marker.line.width = 2
	exitprofit.marker.symbol = 'diamond-open'
	exitprofit.textposition = 'top center'
	exitprofit.mode = 'markers+text'
	exitprofit.textfont = {'family': 'tahoma', 'size': 11, 'color': 'rgba(0, 255, 0, 1)'}
}catch (e) {}

try {
	exitloss = plotDiv.data.filter(x => x.name == "Exit - Loss")[0];
	exitloss.marker.color = 'rgba(255, 0, 0, 1)'
	exitloss.marker.line.width = 2
	exitloss.marker.symbol = 'diamond-open'
	exitloss.textposition = 'top center'
	exitloss.mode = 'markers+text'
	exitloss.textfont = {'family': 'tahoma', 'size': 11, 'color': 'rgba(255, 0, 0, 1)'}
}catch (e) {}

try {	
	tradeentry = plotDiv.data.filter(x => x.name == "Trade entry")[0];
	tradeentry.marker.color = 'rgba(255, 255, 255, 0.8)'
	tradeentry.marker.size = 11
	tradeentry.marker.symbol = "circle-open"
	tradeentry.marker.line.width = 2
	tradeentry.mode = 'markers+text'
	//tradeentry.textposition = 'top right'
	tradeentry.textposition = 'top center'
	//tradeentry.texttemplate = '<br><br>%{text}'
	tradeentry.textfont = {'family': 'tahoma', 'size': 11, 'color': 'white'}
}catch (e) {}

try {
	plotDiv.layout.yaxis4.range = [-10,10];
}catch (e) {}

try {
	plotDiv.layout.yaxis5.range = [-1,1];
}catch (e) {}

try {
	plotDiv.layout.yaxis.domain = [0.67,1]; //[0.605,1]
} catch (e) {}

try {
	plotDiv.layout.yaxis3.domain = [0.35, 0.65]; //[0.3, 0.58];
} catch (e) {}

try {
	plotDiv.layout.yaxis4.domain = [0.12, 0.33]; // [0.11, 0.295];
} catch (e) {}

try {
	plotDiv.layout.yaxis5.domain = [0, 0.1]; // [0, 0.105];
} catch (e) {}

plotDiv.layout.modebar.add = ['v1hovermode', 'toggleSpikeLines'];
plotDiv.layout.dragmode = "pan";
plotDiv.layout.hovermode = "x";

plotDiv.layout.template.layout.font.color = '#949999';
plotDiv.layout.template.layout.paper_bgcolor = '#1b1a1e';
plotDiv.layout.template.layout.plot_bgcolor = '#1b1a1e';

document.body.style.backgroundColor = "#1b1a1e";

plotDiv.layout.xaxis.showgrid = true;
plotDiv.layout.template.layout.xaxis.gridcolor = '#232c33';
plotDiv.layout.template.layout.xaxis.zerolinecolor = '#949999';

plotDiv.layout.template.layout.yaxis.gridcolor = '#232c33';
plotDiv.layout.template.layout.yaxis.zerolinecolor = '#00CCFF60'; //#415061

plotDiv.layout.template.layout.yaxis.zerolinewidth = 1;

try {
	//plotDiv.layout.xaxis.showticklabels = true;
	plotDiv.layout.yaxis.mirror = true;
	plotDiv.layout.yaxis.showline = true;
	plotDiv.layout.yaxis.linecolor = "#5C5C5C"
	plotDiv.layout.xaxis.mirror = true;
	plotDiv.layout.xaxis.showline = false;
	plotDiv.layout.xaxis.linecolor = "#5C5C5C"
	plotDiv.layout.yaxis.linewidth = 1;
	plotDiv.layout.xaxis.linewidth = 1;
	plotDiv.layout.xaxis.spikedash = 'solid'
	plotDiv.layout.yaxis.spikedash = 'solid'
	plotDiv.layout.xaxis.spikecolor = '#FFFFFFB3'
	plotDiv.layout.yaxis.spikecolor = '#FFFFFFB3'
	plotDiv.layout.xaxis.spikethickness = 1
	plotDiv.layout.yaxis.spikethickness = 1
	plotDiv.layout.xaxis.showticklabels = true
} catch (e) {}

try {
	plotDiv.layout.xaxis3.gridcolor = '#232c33';
	plotDiv.layout.yaxis3.gridcolor = '#232c33';
	plotDiv.layout.yaxis3.mirror = true;
	plotDiv.layout.yaxis3.showline = true;
	plotDiv.layout.yaxis3.linecolor = "#5C5C5C"
	//plotDiv.layout.xaxis3.mirror = true;
	//plotDiv.layout.xaxis3.showline = true;
	plotDiv.layout.xaxis3.linecolor = "#5C5C5C"
	plotDiv.layout.yaxis3.linewidth = 1;
	plotDiv.layout.xaxis3.linewidth = 1;
	plotDiv.layout.xaxis3.spikedash = 'solid'
	plotDiv.layout.yaxis3.spikedash = 'solid'
	plotDiv.layout.xaxis3.spikecolor = '#FFFFFFB3'
	plotDiv.layout.yaxis3.spikecolor = '#FFFFFFB3'
	plotDiv.layout.xaxis3.spikethickness = '1'
	plotDiv.layout.yaxis3.spikethickness = '1'
} catch (e) {}

try {
	plotDiv.layout.xaxis4.gridcolor = '#232c33';
	plotDiv.layout.yaxis4.gridcolor = '#232c33';
	plotDiv.layout.yaxis4.mirror = true;
	plotDiv.layout.yaxis4.showline = true;
	plotDiv.layout.yaxis4.linecolor = "#5C5C5C"
	//plotDiv.layout.xaxis4.mirror = true;
	//plotDiv.layout.xaxis4.showline = true;
	plotDiv.layout.xaxis4.linecolor = "#5C5C5C"
	plotDiv.layout.yaxis4.linewidth = 1;
	plotDiv.layout.xaxis4.linewidth = 1;
	plotDiv.layout.xaxis4.spikedash = 'solid'
	plotDiv.layout.yaxis4.spikedash = 'solid'
	plotDiv.layout.xaxis4.spikecolor = '#FFFFFFB3'
	plotDiv.layout.yaxis4.spikecolor = '#FFFFFFB3'
	plotDiv.layout.xaxis4.spikethickness = '1'
	plotDiv.layout.yaxis4.spikethickness = '1'
} catch (e) {}

try {
	plotDiv.layout.xaxis5.gridcolor = '#232c33';
	plotDiv.layout.yaxis5.gridcolor = '#232c33';
	plotDiv.layout.yaxis5.mirror = true;
	plotDiv.layout.yaxis5.showline = true;
	plotDiv.layout.yaxis5.linecolor = "#5C5C5C"
	//plotDiv.layout.xaxis5.mirror = true;
	//plotDiv.layout.xaxis5.showline = true;
	plotDiv.layout.xaxis5.linecolor = "#5C5C5C";
	plotDiv.layout.yaxis5.linewidth = 1;
	plotDiv.layout.xaxis5.linewidth = 1;
	plotDiv.layout.xaxis5.spikedash = 'solid'
	plotDiv.layout.yaxis5.spikedash = 'solid'
	plotDiv.layout.xaxis5.spikecolor = '#FFFFFFB3'
	plotDiv.layout.yaxis5.spikecolor = '#FFFFFFB3'
	plotDiv.layout.xaxis5.spikethickness = '1'
	plotDiv.layout.yaxis5.spikethickness = '1'
	plotDiv.layout.xaxis5.showticklabels = true
} catch (e) {}

for (let i = 1; i < plotDiv.data.length; i++) {
	plotDiv.data[i].visible = "legendonly";
}

//try {
//	plotDiv.data.filter(x => x.name == "true_close")[0].visible = true;
//} catch (e) {}

try {
	document.body.appendChild(document.createElement('script')).src='pair.js';
} catch (e) {}

for (let x = 0; x < plotDiv.data.length; x++) {
	plotDiv.data[x].xaxis = 'x';
	if (plotDiv.data[x].mode == 'lines') {
		plotDiv.data[x].line.width = 1.5;
	}
}

try {
	plotDiv.layout.xaxis.spikemode = 'across+toaxis';
	plotDiv.layout.template.layout.hoverlabel.bgcolor = "000000C1";
	} catch (e) {}

try {
	plotDiv.layout.hovermode = 'x unified';
} catch (e) {}

Plotly.relayout(plotDiv.getAttribute("id"), { annotations: [] });

document.body.style.overflow = 'hidden';
document.body.style.marginTop = -80;
document.body.style.marginBottom = -50;
document.querySelector("div[id*='modebar']").style.top = '81px';

window.dispatchEvent(new Event('resize'));

plotDiv.on('plotly_selected', function(){
	selectioras = []
	for (let x = 0; x < plotDiv.data.length; x++) {
			if (plotDiv.data[x].visible == true) {
				selectioras.push("\n"+plotDiv.data[x].name)
				for (let i = 0; i < plotDiv.data[0].selectedpoints.length; i++) {
					j = plotDiv.data[0].selectedpoints[i]
					selectioras.push(plotDiv.data[x].y[j])
			}
		}
	}
	console.log(selectioras);
});

function download(filename, text) {
	var element = document.createElement('a');
	element.setAttribute('href', 'data:text/plain;charset=utf-8,' + encodeURIComponent(text));
	element.setAttribute('download', filename);
	element.style.display = 'none';
	document.body.appendChild(element);
	element.click();
	document.body.removeChild(element);
}

function handler() {
	download("good.csv",selectioras.toString().replaceAll(',',';'))
}

document.querySelector("[id^='modebar'] > div:nth-child(4) > a").dataset.title = "save big benis information"
document.querySelector("[id^='modebar'] > div:nth-child(4) > a").removeAttribute('href')
document.querySelector("[id^='modebar'] > div:nth-child(4) > a").addEventListener("click", handler);

function zoomer(x,y) {
	try {
	document.querySelector("[class^='hoverlayer']").querySelector("[class^=legend]").setAttribute('transform',`translate(${x+30},${y+100})`)
	document.querySelector("[class^='hoverlayer']").querySelector("[class^=legend]").querySelector("[class^=bg]").setAttribute('style',0)
	} catch (e) {}
}
let mousePos = { x: undefined, y: undefined };

onmousemove = function(event) {
	mousePos = { x: event.clientX, y: event.clientY };
	zoomer(mousePos.x,mousePos.y)
	for (let i = 200; i < 1400; i+=200) 
	{
		setTimeout(function() {
			zoomer(mousePos.x,mousePos.y)
			console.log(xD)},i)
	}
}

// plotDiv.on('plotly_hover',function(){
// 	//console.log(xD)
// 	//var e = e || window.event;
// 	//console.log('Cursor Position: x=' + e.clientX + ', y=' + e.clientY);
// });
