<cluster_comparison>
<!--
  <div style="text-align: center">
    <img id="cluster_comparison" width="75%" height="75%">
  </div>
 -->

<script>
  this.positives = opts.positives;
  this.negatives = opts.negatives;
  this.topn = opts.topn;
  // this.topn = opts.topn;
  this.n_clusters = opts.n_clusters;

  var positives = 'positive[]=' + this.positives.join('&positive[]=');
  var negatives = 'negative[]=' + this.negatives.join('&negative[]=');
  var topn = this.topn;
  var n_clusters = this.n_clusters;
  var url = "/pca_most_similar?" + positives + "&" + negatives + "&topn=" + topn + "&n_clusters=" + n_clusters;
  var image_url = "/plot_cluster_comparison?" + positives + "&" + negatives + "&topn=" + topn + "&n_clusters=" + n_clusters;
  
  d3.select("#cluster_comparison")
    .attr("src", image_url);

</script>
</cluster_comparison>
