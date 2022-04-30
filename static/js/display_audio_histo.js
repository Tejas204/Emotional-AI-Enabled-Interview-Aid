d3.csv("C:/Users/DELL/Work/BE project/prototype/static/js/db/new_audio_emotions_dist.csv").then(makeChart);
function makeChart(data) {
      var emotion = data.map(function(d) {return d.EMOTION;});
      var value = data.map(function(d) {return d.VALUE;});



// Bar chart
new Chart(document.getElementById("audio_emotions_dist"), {
    type: 'bar',
    data: {
      labels: emotion,
      datasets: [
        {
          label: "Value",
          backgroundColor: ["#3e95cd", "#8e5ea2","#3cba9f","#e8c3b9","#c45850", "#3e95cd", "#8e5ea2", "#3cba9f", "e8c3b9"],
          data: value
        }
      ]
    },
    options: {
      legend: { display: false },
      title: {
        display: true,
        text: 'Emotions vs value'
      }
    }
});
}
