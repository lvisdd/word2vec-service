<wordanimations>
   <!--
   <div id="wordanimations" class="form-group">
     <svg></svg>
   </div>
    -->

<!--
 General Update Pattern, III
 https://bl.ocks.org/mbostock/3808234
 https://www.webprofessional.jp/how-react-makes-your-d3-better/
 https://www.sitepoint.com/how-react-makes-your-d3-better/
 -->
<style>
    #wordanimations text {
        font: bold 48px monospace;
    }

    .enter {
        fill: green;
    }

    .update {
        fill: #333;
    }

    .exit {
        fill: brown;
    }
</style>
<script>
    this.positives = opts.positives;
    this.negatives = opts.negatives;
    this.topn = opts.topn;
       
    // var alphabet = "abcdefghijklmnopqrstuvwxyz".split("");
    var alphabet = [];
    var positives = this.positives;
    var negatives = this.negatives;
    
    if(positives) {
        alphabet = alphabet.concat(positives)
    }

    if(negatives) {
        alphabet = alphabet.concat(negatives)
    }
    
    // var width = 960;
    // var height = 500;
    // var width = 600;
    var height = 96 * alphabet.length;
    var width = window.innerWidth
    // var height = window.inneHeight;

    d3.select("#wordanimations").selectAll('svg').remove();
    var svg = d3.select("#wordanimations")
                .append("svg")
                .attr("width", width)
                .attr("height", height);

    // var svg = d3.select("svg"),
    width = +svg.attr("width"),
    height = +svg.attr("height"),
    // g = svg.append("g").attr("transform", "translate(32," + (height / 2) + ")");
    g = svg.append("g")
           .attr("font-weight", "bold")
           // .attr("font-size", "48px")
           .attr("font-size", "24px")
           // .attr("font-family", "monospace")
           // .attr("transform", "translate(32," + (height / 2) + ")");
           // .attr("transform", "translate(48," + (height / 2) + ")");
           // .attr("transform", "translate(24," + (height / 2) + ")");
           // .attr("transform", "translate(24, 24)");
           // .attr("transform", "translate(" + (width / 2) + ", " + 24 + ")");
           // .attr("transform", "translate(" + (width / 2) + ", " + (height / 2) + ")");
           .attr("transform", "translate(" + (width / 4) + ", " + 24 + ")");

    var t = d3.transition()
              .duration(750);
    
    // function update(data) {
    function update(alphabet) {

        var t = d3.transition()
                  .duration(750);

        var data = [];
        for (var i = 0; i < alphabet.length; i++) {
            data[i] = { id: i, value: alphabet[i] };
        }

        // JOIN new data with old elements.
        var text = g.selectAll("text")
                    // .data(data, function (d) { return d; });
                    .data(data, function (d) { return d.id; });

        // EXIT old elements not present in new data.
        text.exit()
            .attr("class", "exit")
            .transition(t)
            // .attr("y", 60)
            .attr("y", 0)
            .style("fill-opacity", 1e-6)
            .remove();

        // UPDATE old elements present in new data.
        text.attr("class", "update")
            // .attr("y", 0)
            // .attr("y", function (d, i) { return i * 24; })
            .attr("y", function (d, i) { return (i + 1) * 48; })
            .style("fill-opacity", 1)
            .transition(t)
            // .attr("x", function (d, i) { return i * 32; });
            // .attr("x", function (d, i) { return i * 48; });
            // .attr("x", function (d, i) { return i * 24; });
            .attr("x", 0);
        
        // ENTER new elements present in new data.
        text.enter().append("text")
            .attr("class", "enter")
            // .attr("dy", ".35em")
            // .attr("dy", ".70em")
            // .attr("y", -60)
            // .attr("y", function (d) { return -60 * (d.id + 1); })
            // .attr("y", 0)
            // .attr("y", function (d, i) { return -60 * (i + 1) + 16; })
            // .attr("x", function (d, i) { return i * 32; })
            // .attr("x", function (d, i) { return i * 48; })
            // .attr("x", function (d, i) { return i * 24; })
            .attr("x", 0)
            .style("fill-opacity", 1e-6)
            // .text(function (d) { return d; })
            .text(function (d) { return d.value; })
            .transition(t)
            // .attr("y", 0)
            // .attr("y", function (d, i) { return i * 24; })
            .attr("y", function (d, i) { return (i + 1) * 48; })
            .style("fill-opacity", 1);
    }

    // The initial display.
    update(alphabet);

    next = "";

    // Grab a random sample of letters from the alphabet, in alphabetical order.
    // d3.interval(function () {
    setInterval(function () {
        update(
            // d3.shuffle(alphabet)
            // .slice(0, Math.floor(Math.random() * 26))
            // .sort()
            // alphabet.slice(0, Math.floor(Math.random() * alphabet.length))
            alphabet.slice(0, next.length)
            );
        if (alphabet.length == next.length) {
            next = "";
        } else {
            next = alphabet.slice(0, next.length + 1);
        }
        // }, 1500);
    }, 1000);

</script>
</wordanimations>
