<most_similar>
  <style scoped="">
    /* valid color */
    .input-field input[type=text][name^="negative[]"].valid {
    border-bottom: 1px solid #ffc107;
    box-shadow: 0 1px 0 0 #ffc107;
    }
  </style>

  <div class="card">
  <div class="card-block">
    <div class="text-xs-center">
      <h3>{ opts.title }</h3>
      <!--
      <h5>/most_similar?word=[keyword]&topn=[number]</h5>
      <h5>ex) /most_similar?word=king&topn=10</h5
      <h5>/most_similar?positive[]=[word1]&positive[]=word2& .. &negative[]=[word1]&negative[]=word2& .. &topn=[number]</h5>
      <h5>ex) /most_similar?positive[]=king&positive[]=woman&negative[]=man&topn=10</h5>
       -->
      <hr class="m-t-2 m-b-2">
    </div>
<!--
    <p class="card-text">/most_similar</p>
    <p class="card-text">?positive[]="word1"&positive[]="word2"& .. </p>
    <p class="card-text">&negative[]="word1"&negative[]="word2"& .. </p>
    <p class="card-text">&topn="number"</p>
    <hr class="m-t-2 m-b-2">
 -->
    <p class="card-text">ex) king + woman - man = queen</p>
    <p class="card-text">/most_similar?positive[]=king&positive[]=woman&negative[]=man&topn=10</p>
    <hr class="m-t-2 m-b-2">

    <form method="post" action="/most_similar/" data-toggle="validator" role="form" autocomplete="off">
      <!-- 
      <div class="md-form">
        <input type="text" id="form1" name="keyword" class="form-control">
        <label for="form1">keyword：</label>
      </div>
       -->
      
      <div class="md-form" each="{ positive, i in positives }">
          <input type="text" id="positive{ i }" name="positive[]" value="{ positive.word }" placeholder="" class="form-control validate" onblur={ removePositives }>
          <label for="positive{ i }" data-error="wrong" data-success="" class="active">positive #{ i + 1 }：</label>
      </div>

      <!--
      <form onsubmit={ addPositives }>
        <div class="md-form">
          <input type="text" id="positive-new" name="input" onkeyup={ editPositives } onblur={ addPositives }>
          <label for="positive-new" data-error="wrong" data-success="" class="active">positive #NEW：</label>

          <button class="btn btn-deep-purple" disabled={ !positive }>
            Add #{ positives.length + 1 }
          </button>

          <button class="btn btn-danger" disabled={ positives.length == 0 } onclick={ removePositives }>
            Delete #{ positives.length }
          </button>
        </div>
      </form>
       -->
     
      <div class="md-form" each="{ negative, i in negatives }">
        <input type="text" id="negative{ i }" name="negative[]" value="{ negative.word }" placeholder="" class="form-control validate" onblur={ removeNegatives }>
        <label for="negative{ i }" data-error="wrong" data-success="" class="active">negative #{ i + 1 }：</label>
      </div>

      <!--
      <form onsubmit={ addNegatives }>
        <div class="md-form">
          <input type="text" id="negative-new"  name="input" onkeyup={ editNegatives } onblur={ addNegatives }>
          <label for="negative-new" data-error="wrong" data-success="" class="active">negative #NEW：</label>

          <button class="btn btn-deep-purple" disabled={ !negative }>
            Add #{ negatives.length + 1 }
          </button>

          <button class="btn btn-danger" disabled={ negatives.length == 0 } onclick={ removeNegatives }>
            Delete #{ negatives.length }
          </button>
        </div>
      </form>
       -->

      <div class="md-form">
        <input type="number" id="topn" name="topn" min="1" max="100" value="{ opts.topn }" required class="form-control validate">
        <label for="topn" data-error="1..100" data-success="">topn：</label>
      </div>

      <hr class="m-t-2 m-b-2">
      
      <div class="text-xs-center">
        <h3>Clustering Algorithms</h3>
        <hr class="m-t-2 m-b-2">
      </div>
      
      <div class="md-form">
        <select id="clustering_name" name="clustering_name" required class="form-control validate" style="height: 45px;">
          <!--
          <option value="" disabled selected>Clustering Algorism</option>
           -->
          <option value="KMeans">KMeans</option>
          <option each="{ clustering, i in clustering_names }">{ clustering.name }</option>
        </select>
        <!--
        <label for="clustering_name" data-error="wrong" data-success="">Clustering Name：</label>
        <label>Clustering Name</label>
         -->
      </div>

      <div class="md-form">
        <input type="number" id="n_clusters" name="n_clusters" min="1" max="10" value="{ opts.n_clusters }" required class="form-control validate">
        <label for="n_clusters" data-error="1..10" data-success="">n_clusters：</label>
      </div>

      <!--
      <div class="btn-group">
        <button class="btn btn-primary dropdown-toggle" type="button" data-toggle="dropdown" aria-haspopup="true" aria-expanded="false">Clustering Algorithm</button>

        <div class="dropdown-menu">
          <a class="dropdown-item" href="#" each="{ clustering, i in clustering_names }">{ clustering.name }</a>
          <div class="dropdown-divider"></div>
        </div>
      </div>
       -->

      <div class="text-xs-center">
      <!--
        <button type="submit" class="btn btn-indigo" disabled={ positives.length == 0 }>Send</button>
       -->
        <button type="submit" class="btn btn-indigo" disabled={ positives.length == 1 }>Send</button>
        <button type="reset" class="btn btn-cyan" onclick={ resetLists }>Reset</button>
      </div>
    </form>
  </div>
</div>

<script>
  this.positives = opts.positives
  this.negatives = opts.negatives
  this.clustering_names = opts.clustering_names
  this.n_clusters = opts.n_clusters

  resetLists(e) {
    this.positive = ''
    this.negative = ''
    this.positives = { word: '' }
    this.negatives = { word: '' }
    // this.update()
    // this.unmount()
    this.unmount(true)
    riot.mount('most_similar', {
        title: 'most_similar',
        positives: [
            { word: '' },
        ],
        negatives: [
            { word: '' },
        ],
        topn: 10,
        clustering_names: [
            { name: 'KMeans' },
            { name: 'MiniBatchKMeans' },
            { name: 'AffinityPropagation' },
            { name: 'MeanShift' },
            { name: 'SpectralClustering' },
            { name: 'Ward' },
            { name: 'AgglomerativeClustering' },
            { name: 'DBSCAN' },
            { name: 'Birch' },
        ],
        n_clusters: 3,
    })
  }
  
  editPositives(e) {
    this.positive = e.target.value
  }

  addPositives(e) {
    if (this.positive) {
      this.positives.push({ word: this.positive })
      this.positive = this.input.value = ''
    }
  }

  removePositives(e) {
    this.positives[e.item.i] = { word: e.target.value}
    // if (!(e.target.value) && (this.positives.length != 1)) {
    if (!(e.target.value)) {
      // this.positives.pop()
      this.positives.splice(e.item.i, 1);
    }
    // if (this.positives[this.positives.length - 1].word != "") {
    //   this.positives.push({ word: "" })
    // }
    this.positives = this.positives.filter(function(v){
      return v.word != "";
    });
    this.positives.push({ word: "" })
  }

  editNegatives(e) {
    this.negative = e.target.value
  }
  
  addNegatives(e) {
    if (this.negative) {
      this.negatives.push({ word: this.negative })
      this.negative = this.input.value = ''
    }
  }
  
  removeNegatives(e) {
    this.negatives[e.item.i] = { word: e.target.value}
    // if (!(e.target.value) && (this.negatives.length != 1)) {
    if (!(e.target.value)) {
      // this.negatives.pop()
      this.negatives.splice(e.item.i, 1);
    }
    // if (this.negatives[this.negatives.length - 1].word != "") {
    //   this.negatives.push({ word: "" })
    // }
    this.negatives = this.negatives.filter(function(v){
      return v.word != "";
    });
    this.negatives.push({ word: "" })
  }

  this.on('mount', function() {
    Materialize.updateTextFields()
  })
</script>

<!-- 
<h3>{ opts.title }</h3>

<ul>
  <li each={ items.filter(whatShow) }>
    <label class={ completed: done }>
      <input type="checkbox" checked={ done } onclick={ parent.toggle }> { title }
    </label>
  </li>
</ul>

<form onsubmit={ add }>
  <input name="input" onkeyup={ edit }>
  <button disabled={ !text }>Add #{ items.filter(whatShow).length + 1 }</button>

  <button disabled={ items.filter(onlyDone).length == 0 } onclick={ removeAllDone }>
  X{ items.filter(onlyDone).length } </button>
</form>
 -->

<!-- this script tag is optional -->
<!-- 
<script>
  this.items = opts.items

  edit(e) {
    this.text = e.target.value
  }

  add(e) {
    if (this.text) {
      this.items.push({ title: this.text })
      this.text = this.input.value = ''
    }
  }

  removeAllDone(e) {
    this.items = this.items.filter(function(item) {
      return !item.done
    })
  }

  // an two example how to filter items on the list
  whatShow(item) {
    return !item.hidden
  }

  onlyDone(item) {
    return item.done
  }

  toggle(e) {
    var item = e.item
    item.done = !item.done
    return true
  }
</script>
 -->
</most_similar>
