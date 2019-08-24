<scatterchart>
  <div id="scatterchart">
    <svg></svg>
  </div>
   
<script>
  this.positives = opts.positives;
  this.negatives = opts.negatives;
  this.topn = opts.topn;
  this.clustering_name = opts.clustering_name;
  this.n_clusters = opts.n_clusters;

  var positives = 'positive[]=' + this.positives.join('&positive[]=');
  var negatives = 'negative[]=' + this.negatives.join('&negative[]=');
  var topn = this.topn;
  var clustering_name = this.clustering_name;
  var n_clusters = this.n_clusters;
  var url = "/pca_most_similar?" + positives + "&" + negatives + "&topn=" + topn + "&clustering_name=" + clustering_name + "&n_clusters=" + n_clusters;
  // var url = "/pca_most_similar?" + positives + "&" + negatives + "&topn=" + topn;

  nv.addGraph(function () {
    var height = 600 + Math.round(topn / 7) * 20;
    var shuffle = function () { return Math.random() - .5 };
    var chart = nv.models.scatterChart()
                  .showDistX(true)
                  .showDistY(true)
                  .transitionDuration(350)
                  // .color(d3.scale.category10().range());
                  .color(d3.scale.category20().range())
                  .height(height);

    //Configure how the tooltip looks.
    chart.tooltipContent(function (key) {
        return '<h3>' + key + '</h3>';
    });

    //Axis settings
    // chart.xAxis.tickFormat(d3.format('.02f'));
    // chart.yAxis.tickFormat(d3.format('.02f'));
    chart.xAxis.tickFormat(d3.format('.06f'));
    chart.yAxis.tickFormat(d3.format('.06f'));

    chart.xDomain([-1, 1]);
    chart.yDomain([-1, 1]);

    //We want to show shapes other than circles.
    // chart.scatter.onlyCircles(false);
    chart.scatter.onlyCircles(true);

    // d3.select('#scatterchart svg')
    //     .datum(data(4, 40))
    //   .transition().duration(500)
    //     .call(chart);

    var words = [];        
    // d3.json(url, function (data) {
    d3.json(url, function (json) {
        // for (var i = 0; i < data.length; i++) {
        //     var word = { key: data[i][0], similarity: data[i][1], url: "/most_similar?word=" + data[i][0] + "&topn={{ topn }}" };
        //     words.push(word);
        // }
        data = json[0].cluster_nodes;
        // console.log(data);

        for (var i = 0; i < data.length; i++) {
            words.push({
                // key: data[i][1],
                key: data[i].key,
                values: []
            });

            // for (j = 0; j < points; j++) {
            words[i].values.push({
                x: data[i].x
              , y: data[i].y
              , size: 10
                // , size: Math.random()
                //, shape: shapes[j % 6]
            });
            // }

        }
        d3.select('#scatterchart svg')
            // .datum(data(4, 40))
            .datum(words)
            // .transition().duration(500)
            .call(chart)
            .attr("height",height); 

        nv.utils.windowResize(chart.update);
        return chart;
    });

    // nv.utils.windowResize(chart.update);
    // return chart;
  });

  function data(groups, points) {
    var data = [],
        shapes = ['circle', 'cross', 'triangle-up', 'triangle-down', 'diamond', 'square'],
        random = d3.random.normal();

    for (i = 0; i < groups; i++) {
        data.push({
            key: 'Group ' + i,
            values: []
        });

        for (j = 0; j < points; j++) {
            data[i].values.push({
                x: random()
            , y: random()
            , size: Math.random()
                //, shape: shapes[j % 6]
            });
        }
    }

    return data;
  }
</script>
</scatterchart>
