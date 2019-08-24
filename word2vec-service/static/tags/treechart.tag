<treechart>
  <div id="treechart"></div>

<script>
    this.positives = opts.positives;
    this.negatives = opts.negatives;
    this.topn = opts.topn;
    
    var positives = 'positive[]=' + this.positives.join('&positive[]=');
    var negatives = 'negative[]=' + this.negatives.join('&negative[]=');
    var topn = this.topn;
    var url = "/most_similar?" + positives + "&" + negatives + "&topn=" + topn;
    
    nv.addGraph(function () {
        // var chart = nv.models.indentedTree()
        var chart = nv.models.indentedTree()
                      .tableClass('table table-striped') //for bootstrap styling
                      .columns([
                        {
                            key: 'key',
                            label: 'Word',
                            showCount: true,
                            width: '75%',
                            type: 'text',
                            classes: function (d) { return d.url ? 'clickable name' : 'name' },
                            click: function (d) {
                                if (d.url) window.location.href = d.url;
                            }
                        },
                        {
                            // key: 'type',
                            key: 'similarity',
                            label: 'Similarity',
                            width: '25%',
                            type: 'text'
                        }
                      ])
                      .iconOpen('/static/images/grey-plus.png')
                      .iconClose('/static/images/grey-minus.png')
        ;

        // // var url = "/most_similar?word={{ keyword }}&topn={{ topn }}";
        // var positives = 'positive[]=' + decodeURI({{ positives | safe}}).split(",").join('&positive[]=');
        // var negatives = 'negative[]=' + decodeURI({{ negatives | safe}}).split(",").join('&negative[]=');
        // var topn = {{ topn }};
        // var url = "/most_similar?" + positives + "&" + negatives + "&topn=" + topn;
        var words = [];
        d3.json(url, function (data) {
            for (var i = 0; i < data.length; i++) {
                // var word = { key: data[i][0], similarity: data[i][1], url: "/most_similar?word=" + data[i][0] + "&topn={{ topn }}" };
                var word = { key: data[i][0], similarity: data[i][1], url: "/most_similar?positive[]=" + data[i][0] + "&topn=" + topn };
                words.push(word);
            }

            // var values = [{ key: "most_similar", _values: words }]
            // var treedata = [{ key: '{{ keyword }}', url: url, values: values }];
            var treedata = [{ key: 'most_similar', url: url, values: words }];
            d3.select('#treechart')
                // .datum(testData())
                .datum(treedata)
              .call(chart);

            return chart;
        });
    });

    /**************************************
     * Example data
     */

    function testData() {
        return [{
            key: 'NVD3',
            url: 'http://novus.github.com/nvd3',
            values: [
              {
                  key: "Charts",
                  _values: [
                    {
                        key: "Simple Line",
                        type: "Historical",
                        url: "http://novus.github.com/nvd3/ghpages/line.html"
                    },
                    {
                        key: "Scatter / Bubble",
                        type: "Snapshot",
                        url: "http://novus.github.com/nvd3/ghpages/scatter.html"
                    },
                    {
                        key: "Stacked / Stream / Expanded Area",
                        type: "Historical",
                        url: "http://novus.github.com/nvd3/ghpages/stackedArea.html"
                    },
                    {
                        key: "Discrete Bar",
                        type: "Snapshot",
                        url: "http://novus.github.com/nvd3/ghpages/discreteBar.html"
                    },
                    {
                        key: "Grouped / Stacked Multi-Bar",
                        type: "Snapshot / Historical",
                        url: "http://novus.github.com/nvd3/ghpages/multiBar.html"
                    },
                    {
                        key: "Horizontal Grouped Bar",
                        type: "Snapshot",
                        url: "http://novus.github.com/nvd3/ghpages/multiBarHorizontal.html"
                    },
                    {
                        key: "Line and Bar Combo",
                        type: "Historical",
                        url: "http://novus.github.com/nvd3/ghpages/linePlusBar.html"
                    },
                    {
                        key: "Cumulative Line",
                        type: "Historical",
                        url: "http://novus.github.com/nvd3/ghpages/cumulativeLine.html"
                    },
                    {
                        key: "Line with View Finder",
                        type: "Historical",
                        url: "http://novus.github.com/nvd3/ghpages/lineWithFocus.html"
                    }
                  ]
              },
              {
                  key: "Chart Components",
                  _values: [
                    {
                        key: "Legend",
                        type: "Universal",
                        url: "http://novus.github.com/nvd3/examples/legend.html"
                    }
                  ]
              }
            ]
        }];
    }
</script>
</treechart>
