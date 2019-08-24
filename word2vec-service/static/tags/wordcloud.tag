<wordcloud>
  <div id="wordcloud" class="form-group"></div>

<!--
 Word Cloud
 https://www.jasondavies.com/wordcloud/
 https://jsfiddle.net/hayakawatomoaki/vc2n5Lg5/2/
 -->

<script>
  this.positives = opts.positives
  this.negatives = opts.negatives
  this.topn = opts.topn
  
  var positives = 'positive[]=' + this.positives.join('&positive[]=');
  var negatives = 'negative[]=' + this.negatives.join('&negative[]=');
  var topn = this.topn;
  var url = "/most_similar?" + positives + "&" + negatives + "&topn=" + topn;
  
  d3.select("#wordcloud").selectAll('svg').remove();
  d3.json(url, function (data) {
    // var countMax = d3.max(data, function (d) { return d.count });
    var countMax = data.length;
    // var sizeScale = d3.scale.linear().domain([0, countMax]).range([10, 100]);
    var sizeScale = d3.scale.linear().domain([countMax, 0]).range([10, 100]);
    var colorScale = d3.scale.category20();
    // var fill = d3.scaleOrdinal(d3.schemeCategory20);

    var words = [];
    // for (var i = (data.length - 1) ; i >= 0; i--) {
    for (var i = 0 ; i < data.length; i++) {
        word = { text: data[i][0], url: "/most_similar?positive[]=" + data[i][0] + "&topn=" + topn, size: (sizeScale(i)) };
        words.push(word);
    }

    var width = 800;
    var height = 600;
    // var width = window.innerWidth;
    // var height = window.innerHeight;
    var gap_length = 0;
    
    d3.layout.cloud()
      .size([width, height])
      .words(words)
      // .padding(5)
      .padding(gap_length)
      // .rotate(function () { return ~~(Math.random() * 2) * 90; })
      .rotate(function () { return Math.round(1 - Math.random()) * 90; })
      .font("Impact")
      .fontSize(function (d) { return d.size; })
      // .fontSize(function (d) { return sizeScale(d.size); })
      .on("end", draw)
      .start();
    
    function draw(words) {
        d3.select("#wordcloud")
          .append("svg")
          .attr("width", width)
          .attr("height", height)
          .append("g")
          // .attr("transform", "translate(" + width + "," + height + ")")
          .attr("transform", "translate(" + [width >> 1, height >> 1] + ")")
          // .attr("transform", "translate(150,150)")
          .selectAll("text")
          .data(words)
          .enter()
          .append("text")
          .style("font-size", function (d) { return d.size + "px"; })
          .style("font-family", "Impact")
          // .style("fill", function (d, i) { return fill(i); })
          .style("fill", function (d, i) { return colorScale(i); })
          .style("cursor", "pointer")
          .attr("text-anchor", "middle")
          .attr("transform", function (d) {
              return "translate(" + [d.x, d.y] + ")rotate(" + d.rotate + ")";
          })
          .text(function (d) { return d.text; })
          .on("click", function (d, i) {
              window.open(d.url, "_blank");
          });
    }

  });
</script>
</wordcloud>
