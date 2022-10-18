var plotDiv = document.getElementsByClassName("plotly-graph-div js-plotly-plot")[0];

plotDiv.data.filter(x => x.name == "Volume");

VolumeShudas = plotDiv.data.filter(x => x.name == "Volume")[0];

delete VolumePls.x;
delete VolumePls.y;
VolumePls.x = "";
VolumePls.y = "";
VolumePls.xaxis = "";
VolumePls.yaxis = "";

try {
	plotDiv.layout.yaxis.domain = [0.605,1];
} catch (e) {}

try {
	plotDiv.layout.yaxis3.domain = [0.3, 0.6];
} catch (e) {}

try {
	plotDiv.layout.yaxis4.domain = [0.11, 0.295];
} catch (e) {}

try {
	plotDiv.layout.yaxis5.domain = [0, 0.105];
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
plotDiv.layout.template.layout.xaxis.zerolinecolor = '#415061';

plotDiv.layout.template.layout.yaxis.gridcolor = '#232c33';
plotDiv.layout.template.layout.yaxis.zerolinecolor = '#415061';

plotDiv.layout.template.layout.yaxis.zerolinewidth = 1;

for (let i = 1; i < plotDiv.data.length; i++) {
	plotDiv.data[i].visible = "legendonly";
}

document.querySelector("div:nth-child(4) > a:nth-child(3)").click();
Plotly.relayout(plotDiv.getAttribute("id"), { annotations: [] });