

```python
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

```


```python
trainDF = pd.read_csv(open("C:\\Users\\Jérémie\\IdeaProjects\\titanic-kaggle-py\\data\\train.csv", 'r'))
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>714.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>446.000000</td>
      <td>0.383838</td>
      <td>2.308642</td>
      <td>29.699118</td>
      <td>0.523008</td>
      <td>0.381594</td>
      <td>32.204208</td>
    </tr>
    <tr>
      <th>std</th>
      <td>257.353842</td>
      <td>0.486592</td>
      <td>0.836071</td>
      <td>14.526497</td>
      <td>1.102743</td>
      <td>0.806057</td>
      <td>49.693429</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.420000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>223.500000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>20.125000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>7.910400</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>446.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>28.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>14.454200</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>668.500000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>38.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>31.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>891.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>80.000000</td>
      <td>8.000000</td>
      <td>6.000000</td>
      <td>512.329200</td>
    </tr>
  </tbody>
</table>
</div>




```python
train_df = trainDF
```

We have the following columns :


```python
train_df.dtypes
```




    PassengerId      int64
    Survived         int64
    Pclass           int64
    Name            object
    Sex             object
    Age            float64
    SibSp            int64
    Parch            int64
    Ticket          object
    Fare           float64
    Cabin           object
    Embarked        object
    dtype: object




```python
train_df.head(10)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>0</td>
      <td>3</td>
      <td>Moran, Mr. James</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>330877</td>
      <td>8.4583</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>0</td>
      <td>1</td>
      <td>McCarthy, Mr. Timothy J</td>
      <td>male</td>
      <td>54.0</td>
      <td>0</td>
      <td>0</td>
      <td>17463</td>
      <td>51.8625</td>
      <td>E46</td>
      <td>S</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>0</td>
      <td>3</td>
      <td>Palsson, Master. Gosta Leonard</td>
      <td>male</td>
      <td>2.0</td>
      <td>3</td>
      <td>1</td>
      <td>349909</td>
      <td>21.0750</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>1</td>
      <td>3</td>
      <td>Johnson, Mrs. Oscar W (Elisabeth Vilhelmina Berg)</td>
      <td>female</td>
      <td>27.0</td>
      <td>0</td>
      <td>2</td>
      <td>347742</td>
      <td>11.1333</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>1</td>
      <td>2</td>
      <td>Nasser, Mrs. Nicholas (Adele Achem)</td>
      <td>female</td>
      <td>14.0</td>
      <td>1</td>
      <td>0</td>
      <td>237736</td>
      <td>30.0708</td>
      <td>NaN</td>
      <td>C</td>
    </tr>
  </tbody>
</table>
</div>



`Name` is a String, a concatenation of last name then first name. It will need some processing.

`Sex` is a qualitative variable.

`Ticket` is quite ugly, as of `Cabin`. Maybe we will work on this at last.

`Embarked` is a qualitative variable.

`Pclass` is an int, but also a qualitative variable.


```python
from bokeh.io import output_notebook, show
output_notebook()
```



    <div class="bk-root">
        <a href="http://bokeh.pydata.org" target="_blank" class="bk-logo bk-logo-small bk-logo-notebook"></a>
        <span id="30c8221c-d9e6-4a3c-ab76-60faf78ca092">Loading BokehJS ...</span>
    </div>





```python
from bokeh.models import ColumnDataSource
cds_df = ColumnDataSource(train_df)
```


```python
from bokeh.charts import Histogram
p = Histogram(train_df["Fare"], palette=['red','blue'])
show(p)
```




    <div class="bk-root">
        <div class="bk-plotdiv" id="fc391734-8a4e-4dc3-a2ed-22dc942287e7"></div>
    </div>
<script type="text/javascript">
  
  (function(global) {
    function now() {
      return new Date();
    }
  
    var force = false;
  
    if (typeof (window._bokeh_onload_callbacks) === "undefined" || force === true) {
      window._bokeh_onload_callbacks = [];
      window._bokeh_is_loading = undefined;
    }
  
  
    
    if (typeof (window._bokeh_timeout) === "undefined" || force === true) {
      window._bokeh_timeout = Date.now() + 0;
      window._bokeh_failed_load = false;
    }
  
    var NB_LOAD_WARNING = {'data': {'text/html':
       "<div style='background-color: #fdd'>\n"+
       "<p>\n"+
       "BokehJS does not appear to have successfully loaded. If loading BokehJS from CDN, this \n"+
       "may be due to a slow or bad network connection. Possible fixes:\n"+
       "</p>\n"+
       "<ul>\n"+
       "<li>re-rerun `output_notebook()` to attempt to load from CDN again, or</li>\n"+
       "<li>use INLINE resources instead, as so:</li>\n"+
       "</ul>\n"+
       "<code>\n"+
       "from bokeh.resources import INLINE\n"+
       "output_notebook(resources=INLINE)\n"+
       "</code>\n"+
       "</div>"}};
  
    function display_loaded() {
      if (window.Bokeh !== undefined) {
        var el = document.getElementById("fc391734-8a4e-4dc3-a2ed-22dc942287e7");
        el.textContent = "BokehJS " + Bokeh.version + " successfully loaded.";
      } else if (Date.now() < window._bokeh_timeout) {
        setTimeout(display_loaded, 100)
      }
    }
  
    function run_callbacks() {
      window._bokeh_onload_callbacks.forEach(function(callback) { callback() });
      delete window._bokeh_onload_callbacks
      console.info("Bokeh: all callbacks have finished");
    }
  
    function load_libs(js_urls, callback) {
      window._bokeh_onload_callbacks.push(callback);
      if (window._bokeh_is_loading > 0) {
        console.log("Bokeh: BokehJS is being loaded, scheduling callback at", now());
        return null;
      }
      if (js_urls == null || js_urls.length === 0) {
        run_callbacks();
        return null;
      }
      console.log("Bokeh: BokehJS not loaded, scheduling load and callback at", now());
      window._bokeh_is_loading = js_urls.length;
      for (var i = 0; i < js_urls.length; i++) {
        var url = js_urls[i];
        var s = document.createElement('script');
        s.src = url;
        s.async = false;
        s.onreadystatechange = s.onload = function() {
          window._bokeh_is_loading--;
          if (window._bokeh_is_loading === 0) {
            console.log("Bokeh: all BokehJS libraries loaded");
            run_callbacks()
          }
        };
        s.onerror = function() {
          console.warn("failed to load library " + url);
        };
        console.log("Bokeh: injecting script tag for BokehJS library: ", url);
        document.getElementsByTagName("head")[0].appendChild(s);
      }
    };var element = document.getElementById("fc391734-8a4e-4dc3-a2ed-22dc942287e7");
    if (element == null) {
      console.log("Bokeh: ERROR: autoload.js configured with elementid 'fc391734-8a4e-4dc3-a2ed-22dc942287e7' but no matching script tag was found. ")
      return false;
    }
  
    var js_urls = [];
  
    var inline_js = [
      function(Bokeh) {
        (function() {
          var fn = function() {
            var docs_json = {"e5d6f7ce-715d-43c8-9c24-63407046fd51":{"roots":{"references":[{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(153.219948, 158.008071]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[3.0],"label":["(153.219948, 158.008071]"],"line_alpha":[1.0],"line_color":["black"],"width":[4.788123364485983],"x":["155.6140095"],"y":[1.5]}},"id":"82002ff1-2455-46dc-9938-89454e038447","type":"ColumnDataSource"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(119.703084, 124.491207]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[4.0],"label":["(119.703084, 124.491207]"],"line_alpha":[1.0],"line_color":["black"],"width":[4.788123364485983],"x":["122.09714550000001"],"y":[2.0]}},"id":"66d25f67-d2d3-41e5-b9f6-b05c02c9caea","type":"ColumnDataSource"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"9361a9ac-11ff-41f6-b306-f51ec931c460","type":"Rect"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(124.491207, 129.279331]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[0.0],"label":["(124.491207, 129.279331]"],"line_alpha":[1.0],"line_color":["black"],"width":[4.788123364485969],"x":["126.88526900000001"],"y":[0.0]}},"id":"9e40ded1-abe9-4359-b6f3-79f9a7eaaef7","type":"ColumnDataSource"},{"attributes":{"data_source":{"id":"66d25f67-d2d3-41e5-b9f6-b05c02c9caea","type":"ColumnDataSource"},"glyph":{"id":"a1bb878c-be38-4795-b906-e574a87d415b","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"e02fa3ba-bf22-459e-80f9-baf2ac6241d7","type":"GlyphRenderer"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"acfd5e48-d70e-4e25-951a-da67c42f65b8","type":"Rect"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(129.279331, 134.067454]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[2.0],"label":["(129.279331, 134.067454]"],"line_alpha":[1.0],"line_color":["black"],"width":[4.788123364485983],"x":["131.6733925"],"y":[1.0]}},"id":"c6497bed-6276-431d-9349-b2b1448d6a9d","type":"ColumnDataSource"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"1a5dc582-bcb9-4a47-b87c-c75e9e7aa22b","type":"Rect"},{"attributes":{"data_source":{"id":"2dc825d1-2e14-4468-8529-4a326bf00ea6","type":"ColumnDataSource"},"glyph":{"id":"3429f133-dcbf-41a4-ae1a-ba7d92b9127e","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"1df48c7a-0d93-4ef4-add2-9c14feb1b3b7","type":"GlyphRenderer"},{"attributes":{"data_source":{"id":"9e40ded1-abe9-4359-b6f3-79f9a7eaaef7","type":"ColumnDataSource"},"glyph":{"id":"9361a9ac-11ff-41f6-b306-f51ec931c460","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"0ed1fd86-8126-4724-ad50-2a6d4dc4bfa6","type":"GlyphRenderer"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"a40af76b-7d9c-4769-81bf-3946b0c8aafb","type":"Rect"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(134.067454, 138.855578]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[5.0],"label":["(134.067454, 138.855578]"],"line_alpha":[1.0],"line_color":["black"],"width":[4.788123364485983],"x":["136.46151600000002"],"y":[2.5]}},"id":"79198644-f4e3-46b8-ab76-ddcf0b845c5f","type":"ColumnDataSource"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(143.643701, 148.431824]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[2.0],"label":["(143.643701, 148.431824]"],"line_alpha":[1.0],"line_color":["black"],"width":[4.788123364485983],"x":["146.03776249999999"],"y":[1.0]}},"id":"2dc825d1-2e14-4468-8529-4a326bf00ea6","type":"ColumnDataSource"},{"attributes":{"data_source":{"id":"c6497bed-6276-431d-9349-b2b1448d6a9d","type":"ColumnDataSource"},"glyph":{"id":"acfd5e48-d70e-4e25-951a-da67c42f65b8","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"b2dfae73-1f11-4627-92dd-194b6e5e6299","type":"GlyphRenderer"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(210.677428, 215.465551]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[4.0],"label":["(210.677428, 215.465551]"],"line_alpha":[1.0],"line_color":["black"],"width":[4.788123364485983],"x":["213.07148949999998"],"y":[2.0]}},"id":"62e2a39f-a116-4db9-82cf-bd8e2dc35930","type":"ColumnDataSource"},{"attributes":{"data_source":{"id":"79198644-f4e3-46b8-ab76-ddcf0b845c5f","type":"ColumnDataSource"},"glyph":{"id":"a40af76b-7d9c-4769-81bf-3946b0c8aafb","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"17320b76-6206-4424-aba5-35bddc1c8782","type":"GlyphRenderer"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"a851e427-7101-4da1-bfbb-27be3eae3f26","type":"Rect"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(148.431824, 153.219948]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[4.0],"label":["(148.431824, 153.219948]"],"line_alpha":[1.0],"line_color":["black"],"width":[4.788123364485983],"x":["150.825886"],"y":[2.0]}},"id":"45dc33d7-d73c-4b87-aecc-053ae58c35de","type":"ColumnDataSource"},{"attributes":{"data_source":{"id":"f2f77dc8-101b-4827-8cdd-d90d7f05cc7b","type":"ColumnDataSource"},"glyph":{"id":"19c37a1b-8dea-4056-bc37-70715f7a2d9f","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"dad68869-da6d-4413-82c3-c83289e53429","type":"GlyphRenderer"},{"attributes":{"data_source":{"id":"45dc33d7-d73c-4b87-aecc-053ae58c35de","type":"ColumnDataSource"},"glyph":{"id":"a851e427-7101-4da1-bfbb-27be3eae3f26","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"504a9cd4-8fd2-4d55-879b-3a01a4d5fa26","type":"GlyphRenderer"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"aabc3892-871f-4d5a-ad04-3b0bff50cc4e","type":"Rect"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(158.008071, 162.796194]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[0.0],"label":["(158.008071, 162.796194]"],"line_alpha":[1.0],"line_color":["black"],"width":[4.788123364485983],"x":["160.4021325"],"y":[0.0]}},"id":"3f43e407-f9ab-4e15-bdcd-7903234ac74b","type":"ColumnDataSource"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"651022c5-b8f5-47f5-b62b-11aa3e56202d","type":"Rect"},{"attributes":{"data_source":{"id":"62e2a39f-a116-4db9-82cf-bd8e2dc35930","type":"ColumnDataSource"},"glyph":{"id":"91cf442d-ffab-43f4-8e18-7989414085a6","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"4635ae54-db50-4100-9609-b79ca57ae1db","type":"GlyphRenderer"},{"attributes":{"data_source":{"id":"82002ff1-2455-46dc-9938-89454e038447","type":"ColumnDataSource"},"glyph":{"id":"1a5dc582-bcb9-4a47-b87c-c75e9e7aa22b","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"d269632c-8a99-4e7b-892e-261d38f85a7a","type":"GlyphRenderer"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"05a7b60c-e8ec-4151-9623-67a4a5cb8414","type":"Rect"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(162.796194, 167.584318]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[2.0],"label":["(162.796194, 167.584318]"],"line_alpha":[1.0],"line_color":["black"],"width":[4.788123364485983],"x":["165.190256"],"y":[1.0]}},"id":"2926fcae-4f92-4527-86be-e2be53e25b14","type":"ColumnDataSource"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"afdf05a7-c7e0-4cc5-9465-a49e1f660f83","type":"Rect"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"91cf442d-ffab-43f4-8e18-7989414085a6","type":"Rect"},{"attributes":{"data_source":{"id":"3f43e407-f9ab-4e15-bdcd-7903234ac74b","type":"ColumnDataSource"},"glyph":{"id":"aabc3892-871f-4d5a-ad04-3b0bff50cc4e","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"2c7b3d26-80d8-4d25-ba8f-6c8d35458635","type":"GlyphRenderer"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"931df527-f893-4dd5-81de-d251969512de","type":"Rect"},{"attributes":{"data_source":{"id":"5bfe4a61-e80f-4468-aa24-d070bbd7f1cd","type":"ColumnDataSource"},"glyph":{"id":"931df527-f893-4dd5-81de-d251969512de","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"7dc3a134-b76f-4404-b3fc-c9a854aa18cd","type":"GlyphRenderer"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"19c37a1b-8dea-4056-bc37-70715f7a2d9f","type":"Rect"},{"attributes":{"data_source":{"id":"2926fcae-4f92-4527-86be-e2be53e25b14","type":"ColumnDataSource"},"glyph":{"id":"05a7b60c-e8ec-4151-9623-67a4a5cb8414","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"2927c6c0-3b2c-4430-a511-76d9e6b4098b","type":"GlyphRenderer"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(167.584318, 172.372441]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[0.0],"label":["(167.584318, 172.372441]"],"line_alpha":[1.0],"line_color":["black"],"width":[4.788123364485983],"x":["169.97837950000002"],"y":[0.0]}},"id":"5bfe4a61-e80f-4468-aa24-d070bbd7f1cd","type":"ColumnDataSource"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(177.160564, 181.948688]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[0.0],"label":["(177.160564, 181.948688]"],"line_alpha":[1.0],"line_color":["black"],"width":[4.788123364485983],"x":["179.55462599999998"],"y":[0.0]}},"id":"f2f77dc8-101b-4827-8cdd-d90d7f05cc7b","type":"ColumnDataSource"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(469.236090, 474.024213]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[0.0],"label":["(469.236090, 474.024213]"],"line_alpha":[1.0],"line_color":["black"],"width":[4.788123364485955],"x":["471.6301515"],"y":[0.0]}},"id":"b322351b-cba4-4a39-9233-f2591e2d8bba","type":"ColumnDataSource"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(186.736811, 191.524935]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[0.0],"label":["(186.736811, 191.524935]"],"line_alpha":[1.0],"line_color":["black"],"width":[4.788123364485983],"x":["189.130873"],"y":[0.0]}},"id":"e6790bb9-e516-4f2d-85a8-cf25521d4ffe","type":"ColumnDataSource"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(344.744882, 349.533006]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[0.0],"label":["(344.744882, 349.533006]"],"line_alpha":[1.0],"line_color":["black"],"width":[4.788123364485955],"x":["347.13894400000004"],"y":[0.0]}},"id":"7a6af713-9672-4127-929a-dcb8b1d04eac","type":"ColumnDataSource"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"de7c0f30-6bd5-4878-b4b0-658780bf65f5","type":"Rect"},{"attributes":{"data_source":{"id":"4988fc2d-c2fa-4a0a-84c4-5ca797ba6fad","type":"ColumnDataSource"},"glyph":{"id":"90278d7b-f794-4824-84ce-703924e7bc1b","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"3b9753aa-eee4-4931-a1c2-bbeb3dd66f01","type":"GlyphRenderer"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"04a3c8d2-efca-4bde-9781-e4ac28e60cdd","type":"Rect"},{"attributes":{"data_source":{"id":"31cfa1df-4c52-41ed-a8d6-bb791fee0b03","type":"ColumnDataSource"},"glyph":{"id":"884630e1-e820-4e8d-a210-d3ed7bf8e12b","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"41f503d9-58d5-43ef-b60b-326cff46bc16","type":"GlyphRenderer"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(181.948688, 186.736811]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[0.0],"label":["(181.948688, 186.736811]"],"line_alpha":[1.0],"line_color":["black"],"width":[4.788123364485983],"x":["184.3427495"],"y":[0.0]}},"id":"a4405cdf-66d9-4500-9617-d1769b109efa","type":"ColumnDataSource"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"37eb6f46-07df-4ce9-9ec1-91e2fcd16a2e","type":"Rect"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"60b4b0af-08e4-4b60-92a2-2ff1fea32b5b","type":"Rect"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"f51c95df-b749-4794-b067-e7ec84a8229e","type":"Rect"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(349.533006, 354.321129]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[0.0],"label":["(349.533006, 354.321129]"],"line_alpha":[1.0],"line_color":["black"],"width":[4.788123364486012],"x":["351.9270675"],"y":[0.0]}},"id":"a57cba9d-1c7a-4ae0-b44c-b78ac37701e4","type":"ColumnDataSource"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(474.024213, 478.812336]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[0.0],"label":["(474.024213, 478.812336]"],"line_alpha":[1.0],"line_color":["black"],"width":[4.788123364486012],"x":["476.4182745"],"y":[0.0]}},"id":"d0d939f0-c62c-4251-85f0-8a5a5eea54c1","type":"ColumnDataSource"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(172.372441, 177.160564]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[0.0],"label":["(172.372441, 177.160564]"],"line_alpha":[1.0],"line_color":["black"],"width":[4.788123364485983],"x":["174.7665025"],"y":[0.0]}},"id":"bdf707d7-5868-4f80-a4fa-df15d378db72","type":"ColumnDataSource"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"fbb1b3fc-2aaa-43e2-b649-b2e3c56523ff","type":"Rect"},{"attributes":{"data_source":{"id":"bdf707d7-5868-4f80-a4fa-df15d378db72","type":"ColumnDataSource"},"glyph":{"id":"04a3c8d2-efca-4bde-9781-e4ac28e60cdd","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"fcaaa70e-ff8e-4caa-afe2-13543dcc3cea","type":"GlyphRenderer"},{"attributes":{"data_source":{"id":"a4405cdf-66d9-4500-9617-d1769b109efa","type":"ColumnDataSource"},"glyph":{"id":"651022c5-b8f5-47f5-b62b-11aa3e56202d","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"65af37c5-f956-495b-bbb9-27ed0d386670","type":"GlyphRenderer"},{"attributes":{"data_source":{"id":"71d2916f-83f2-40b8-b2c4-9ab478354e39","type":"ColumnDataSource"},"glyph":{"id":"662728eb-8165-45bf-a311-f3cbd0d652d0","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"cec476f8-4b39-48ac-802b-e7ea6451df31","type":"GlyphRenderer"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(363.897376, 368.685499]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[0.0],"label":["(363.897376, 368.685499]"],"line_alpha":[1.0],"line_color":["black"],"width":[4.788123364485955],"x":["366.29143750000003"],"y":[0.0]}},"id":"f6ccb773-9a33-483c-9cad-1c82a695262c","type":"ColumnDataSource"},{"attributes":{"data_source":{"id":"7a6af713-9672-4127-929a-dcb8b1d04eac","type":"ColumnDataSource"},"glyph":{"id":"37eb6f46-07df-4ce9-9ec1-91e2fcd16a2e","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"d9f0ef5a-2ec3-4241-9aea-2589372cc9a6","type":"GlyphRenderer"},{"attributes":{"data_source":{"id":"b322351b-cba4-4a39-9233-f2591e2d8bba","type":"ColumnDataSource"},"glyph":{"id":"de7c0f30-6bd5-4878-b4b0-658780bf65f5","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"c5e9547f-11d0-4f9d-88bc-a7a3b4337ff4","type":"GlyphRenderer"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(359.109252, 363.897376]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[0.0],"label":["(359.109252, 363.897376]"],"line_alpha":[1.0],"line_color":["black"],"width":[4.788123364486012],"x":["361.50331400000005"],"y":[0.0]}},"id":"b44bf5c6-dd16-476e-9fc9-445f4fa8711c","type":"ColumnDataSource"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"fbf4fd11-2e37-4b49-b5de-96d78ac18fb5","type":"Rect"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"58cb1ad2-ff68-4a6c-a969-3393c0616b6b","type":"Rect"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(478.812336, 483.600460]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[0.0],"label":["(478.812336, 483.600460]"],"line_alpha":[1.0],"line_color":["black"],"width":[4.788123364485955],"x":["481.20639800000004"],"y":[0.0]}},"id":"9049b169-326c-4af2-9679-15a663e2b33b","type":"ColumnDataSource"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(191.524935, 196.313058]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[0.0],"label":["(191.524935, 196.313058]"],"line_alpha":[1.0],"line_color":["black"],"width":[4.788123364485955],"x":["193.9189965"],"y":[0.0]}},"id":"1436b9b8-3bb0-4534-9b24-9f64f82543b2","type":"ColumnDataSource"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"df9b7fc9-6f08-44af-8d5c-419e591fd651","type":"Rect"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"cab62bce-b26b-4d46-8b0a-c7d490cbaebb","type":"Rect"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"f3b1893c-7b8a-4a50-a0bc-0263adadd8c2","type":"Rect"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(464.447966, 469.236090]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[0.0],"label":["(464.447966, 469.236090]"],"line_alpha":[1.0],"line_color":["black"],"width":[4.788123364486012],"x":["466.842028"],"y":[0.0]}},"id":"31cfa1df-4c52-41ed-a8d6-bb791fee0b03","type":"ColumnDataSource"},{"attributes":{"data_source":{"id":"bd53563d-12dc-4b10-b131-332b3d2a5da3","type":"ColumnDataSource"},"glyph":{"id":"f7fbd96a-6d29-4c6f-81d0-5b733be00bbd","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"b954ada1-1227-4367-aa0b-178969c148eb","type":"GlyphRenderer"},{"attributes":{"data_source":{"id":"e6790bb9-e516-4f2d-85a8-cf25521d4ffe","type":"ColumnDataSource"},"glyph":{"id":"df9b7fc9-6f08-44af-8d5c-419e591fd651","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"5b7a7975-3b07-44a5-a8ef-2aded03017b8","type":"GlyphRenderer"},{"attributes":{"data_source":{"id":"a57cba9d-1c7a-4ae0-b44c-b78ac37701e4","type":"ColumnDataSource"},"glyph":{"id":"f51c95df-b749-4794-b067-e7ec84a8229e","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"e85896bc-96e5-43d2-a582-d0747aa282e0","type":"GlyphRenderer"},{"attributes":{"data_source":{"id":"d0d939f0-c62c-4251-85f0-8a5a5eea54c1","type":"ColumnDataSource"},"glyph":{"id":"60b4b0af-08e4-4b60-92a2-2ff1fea32b5b","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"3a77f097-323a-4090-9bed-ea29b4e74493","type":"GlyphRenderer"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"34f8f174-78a7-4b74-bd12-6cc87b6a5b62","type":"Rect"},{"attributes":{"data_source":{"id":"2c7e26f0-d159-4809-a8de-5bbe1e6a1b05","type":"ColumnDataSource"},"glyph":{"id":"0b35734e-f34e-476b-b959-9ddf5451c58e","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"a9a93c47-b850-4814-9567-408ecf509975","type":"GlyphRenderer"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"6c61a403-96c8-4976-b41f-c7f46e42a1c8","type":"Rect"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(483.600460, 488.388583]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[0.0],"label":["(483.600460, 488.388583]"],"line_alpha":[1.0],"line_color":["black"],"width":[4.788123364486012],"x":["485.9945215"],"y":[0.0]}},"id":"b4354b91-2ed0-4dca-b03c-a17163d837f3","type":"ColumnDataSource"},{"attributes":{"data_source":{"id":"b44bf5c6-dd16-476e-9fc9-445f4fa8711c","type":"ColumnDataSource"},"glyph":{"id":"6c10ad25-5f59-4f68-9289-a4ac7c186cad","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"22d42218-4eeb-4560-ba6f-fcb471b24c43","type":"GlyphRenderer"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(196.313058, 201.101181]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[0.0],"label":["(196.313058, 201.101181]"],"line_alpha":[1.0],"line_color":["black"],"width":[4.788123364485983],"x":["198.7071195"],"y":[0.0]}},"id":"641bbdca-d23c-4d6e-9203-1ab2879ae867","type":"ColumnDataSource"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"6d535ee7-8a7e-44c3-a999-c8245c5a4029","type":"Rect"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(368.685499, 373.473622]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[0.0],"label":["(368.685499, 373.473622]"],"line_alpha":[1.0],"line_color":["black"],"width":[4.788123364486012],"x":["371.07956049999996"],"y":[0.0]}},"id":"e867b0c0-0f16-4f39-a606-e930bf36928d","type":"ColumnDataSource"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(316.016142, 320.804265]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[0.0],"label":["(316.016142, 320.804265]"],"line_alpha":[1.0],"line_color":["black"],"width":[4.788123364485955],"x":["318.41020349999997"],"y":[0.0]}},"id":"2130cdc3-5a6b-409a-a25f-b7ca5365b17d","type":"ColumnDataSource"},{"attributes":{"data_source":{"id":"1436b9b8-3bb0-4534-9b24-9f64f82543b2","type":"ColumnDataSource"},"glyph":{"id":"58cb1ad2-ff68-4a6c-a969-3393c0616b6b","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"8cbff69a-867b-420e-adb8-8e0b0a290510","type":"GlyphRenderer"},{"attributes":{"data_source":{"id":"9049b169-326c-4af2-9679-15a663e2b33b","type":"ColumnDataSource"},"glyph":{"id":"fbf4fd11-2e37-4b49-b5de-96d78ac18fb5","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"5010eb1b-56b9-43d2-b867-27c54c642c21","type":"GlyphRenderer"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"bbd3b33c-4cf6-4236-9c33-d5d2b8a62dcb","type":"Rect"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"cc7755ed-0ed3-4c0b-983b-b7d739a6b142","type":"Rect"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(488.388583, 493.176707]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[0.0],"label":["(488.388583, 493.176707]"],"line_alpha":[1.0],"line_color":["black"],"width":[4.788123364485955],"x":["490.782645"],"y":[0.0]}},"id":"7b96642f-69fa-49ca-aa2b-18fe63db0470","type":"ColumnDataSource"},{"attributes":{"data_source":{"id":"f6ccb773-9a33-483c-9cad-1c82a695262c","type":"ColumnDataSource"},"glyph":{"id":"cab62bce-b26b-4d46-8b0a-c7d490cbaebb","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"73fc26dd-106a-4562-acc7-3eb2dbb00906","type":"GlyphRenderer"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(201.101181, 205.889305]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[0.0],"label":["(201.101181, 205.889305]"],"line_alpha":[1.0],"line_color":["black"],"width":[4.788123364485983],"x":["203.49524300000002"],"y":[0.0]}},"id":"f5042746-3821-4f4f-a9de-67aea5e580d4","type":"ColumnDataSource"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"49f53b3d-2559-49a0-a80a-a311c7bd8e76","type":"Rect"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(373.473622, 378.261746]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[0.0],"label":["(373.473622, 378.261746]"],"line_alpha":[1.0],"line_color":["black"],"width":[4.788123364485955],"x":["375.867684"],"y":[0.0]}},"id":"4f73ad1d-321f-42f0-ba60-671a754baad5","type":"ColumnDataSource"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(502.752953, 507.541077]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[0.0],"label":["(502.752953, 507.541077]"],"line_alpha":[1.0],"line_color":["black"],"width":[4.788123364486012],"x":["505.147015"],"y":[0.0]}},"id":"d328d6df-f048-4342-8edf-8bdc5b14eba8","type":"ColumnDataSource"},{"attributes":{"data_source":{"id":"641bbdca-d23c-4d6e-9203-1ab2879ae867","type":"ColumnDataSource"},"glyph":{"id":"6c61a403-96c8-4976-b41f-c7f46e42a1c8","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"f05da917-b54e-4d65-a40b-1ed8a0f6e3f9","type":"GlyphRenderer"},{"attributes":{"data_source":{"id":"b4354b91-2ed0-4dca-b03c-a17163d837f3","type":"ColumnDataSource"},"glyph":{"id":"34f8f174-78a7-4b74-bd12-6cc87b6a5b62","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"b7143d08-3fbc-45d4-aa96-16f93885ee6c","type":"GlyphRenderer"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"b1212033-047e-4f06-b19c-b52789d5dea6","type":"Rect"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"2c473fd7-0de2-4ed8-a5a2-edd6b5651bc7","type":"Rect"},{"attributes":{"data_source":{"id":"e867b0c0-0f16-4f39-a606-e930bf36928d","type":"ColumnDataSource"},"glyph":{"id":"6d535ee7-8a7e-44c3-a999-c8245c5a4029","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"224b2780-e10f-45e4-8ccf-f150523cdd45","type":"GlyphRenderer"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(205.889305, 210.677428]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[0.0],"label":["(205.889305, 210.677428]"],"line_alpha":[1.0],"line_color":["black"],"width":[4.788123364485983],"x":["208.2833665"],"y":[0.0]}},"id":"11055fa0-be56-4dd2-894d-9b312be33008","type":"ColumnDataSource"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"662728eb-8165-45bf-a311-f3cbd0d652d0","type":"Rect"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"34e615f3-3548-4c4f-8ae3-1c71b0696941","type":"Rect"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(378.261746, 383.049869]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[0.0],"label":["(378.261746, 383.049869]"],"line_alpha":[1.0],"line_color":["black"],"width":[4.788123364486012],"x":["380.65580750000004"],"y":[0.0]}},"id":"a2dc070b-fa34-4242-b01b-f852ad366213","type":"ColumnDataSource"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"c004703c-ac7a-4852-b4be-f84a33ff02f2","type":"Rect"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(220.253675, 225.041798]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[1.0],"label":["(220.253675, 225.041798]"],"line_alpha":[1.0],"line_color":["black"],"width":[4.788123364485983],"x":["222.6477365"],"y":[0.5]}},"id":"84266960-5a18-4008-ae9e-75e28e9381f3","type":"ColumnDataSource"},{"attributes":{"data_source":{"id":"f5042746-3821-4f4f-a9de-67aea5e580d4","type":"ColumnDataSource"},"glyph":{"id":"cc7755ed-0ed3-4c0b-983b-b7d739a6b142","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"04610b76-6c3c-4f86-828a-58ee394e4b4b","type":"GlyphRenderer"},{"attributes":{"data_source":{"id":"7b96642f-69fa-49ca-aa2b-18fe63db0470","type":"ColumnDataSource"},"glyph":{"id":"bbd3b33c-4cf6-4236-9c33-d5d2b8a62dcb","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"dd3988f5-9902-4f87-ae09-1578ccfa9382","type":"GlyphRenderer"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"0b35734e-f34e-476b-b959-9ddf5451c58e","type":"Rect"},{"attributes":{"data_source":{"id":"bb9e298f-43dc-4b56-b0b0-d7916beae080","type":"ColumnDataSource"},"glyph":{"id":"d1bc16a0-0dec-4225-9abc-d0fd8612fa2c","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"7a10a4ea-f93f-4d28-ae3b-e610f065c21a","type":"GlyphRenderer"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(507.541077, 512.329200]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[3.0],"label":["(507.541077, 512.329200]"],"line_alpha":[1.0],"line_color":["black"],"width":[4.788123364485955],"x":["509.9351385"],"y":[1.5]}},"id":"7e9b1595-b118-4aa2-a466-fd911dce7acd","type":"ColumnDataSource"},{"attributes":{"data_source":{"id":"4f73ad1d-321f-42f0-ba60-671a754baad5","type":"ColumnDataSource"},"glyph":{"id":"49f53b3d-2559-49a0-a80a-a311c7bd8e76","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"3ca91b0b-4174-42fd-be45-77659c95ce53","type":"GlyphRenderer"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"f7fbd96a-6d29-4c6f-81d0-5b733be00bbd","type":"Rect"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"0bbe18c7-2ac9-479b-a1a8-ed5bb982c779","type":"Rect"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(383.049869, 387.837993]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[0.0],"label":["(383.049869, 387.837993]"],"line_alpha":[1.0],"line_color":["black"],"width":[4.788123364485955],"x":["385.443931"],"y":[0.0]}},"id":"c62cb59d-7624-4c33-b9d1-a60edbb8f71e","type":"ColumnDataSource"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(497.964830, 502.752953]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[0.0],"label":["(497.964830, 502.752953]"],"line_alpha":[1.0],"line_color":["black"],"width":[4.788123364485955],"x":["500.3588915"],"y":[0.0]}},"id":"71d2916f-83f2-40b8-b2c4-9ab478354e39","type":"ColumnDataSource"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"1f6e0152-8147-41fe-b582-da7bfdf4eea7","type":"Rect"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"10b25519-3529-4c1b-9183-33291cece321","type":"Rect"},{"attributes":{"data_source":{"id":"11055fa0-be56-4dd2-894d-9b312be33008","type":"ColumnDataSource"},"glyph":{"id":"b1212033-047e-4f06-b19c-b52789d5dea6","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"fff21382-b33c-47f1-9146-e747551fdb87","type":"GlyphRenderer"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(421.354856, 426.142979]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[0.0],"label":["(421.354856, 426.142979]"],"line_alpha":[1.0],"line_color":["black"],"width":[4.788123364486012],"x":["423.7489175"],"y":[0.0]}},"id":"2c7e26f0-d159-4809-a8de-5bbe1e6a1b05","type":"ColumnDataSource"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(225.041798, 229.829921]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[4.0],"label":["(225.041798, 229.829921]"],"line_alpha":[1.0],"line_color":["black"],"width":[4.788123364485983],"x":["227.4358595"],"y":[2.0]}},"id":"891ad0c3-67a6-4447-8025-f1aeb9f77329","type":"ColumnDataSource"},{"attributes":{"data_source":{"id":"a2dc070b-fa34-4242-b01b-f852ad366213","type":"ColumnDataSource"},"glyph":{"id":"34e615f3-3548-4c4f-8ae3-1c71b0696941","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"03383d13-6de8-4d05-95f8-fc4afbb7e6fe","type":"GlyphRenderer"},{"attributes":{"data_source":{"id":"d328d6df-f048-4342-8edf-8bdc5b14eba8","type":"ColumnDataSource"},"glyph":{"id":"c004703c-ac7a-4852-b4be-f84a33ff02f2","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"8de9b562-8388-412b-bceb-a9678f96283f","type":"GlyphRenderer"},{"attributes":{"data_source":{"id":"8f76e89c-9442-4dbe-ae5e-a02b0125fc97","type":"ColumnDataSource"},"glyph":{"id":"209a56d5-8b21-4c8a-90aa-450c372c5915","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"3905cb26-8d9e-4248-a882-f0ad151f4fcf","type":"GlyphRenderer"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"d1bc16a0-0dec-4225-9abc-d0fd8612fa2c","type":"Rect"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(215.465551, 220.253675]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[0.0],"label":["(215.465551, 220.253675]"],"line_alpha":[1.0],"line_color":["black"],"width":[4.788123364485983],"x":["217.859613"],"y":[0.0]}},"id":"bd53563d-12dc-4b10-b131-332b3d2a5da3","type":"ColumnDataSource"},{"attributes":{"axis_label":"Fare","formatter":{"id":"8f72211a-b696-4037-b0dd-67d86ef9ef1c","type":"BasicTickFormatter"},"plot":{"id":"69634d55-8bb5-4411-9dcb-fa5caf7293df","subtype":"Chart","type":"Plot"},"ticker":{"id":"74154beb-0beb-4edc-901c-bf7156feae79","type":"BasicTicker"}},"id":"44885f48-29b5-4a4f-8863-1f592e481d3a","type":"LinearAxis"},{"attributes":{"data_source":{"id":"6198d6d6-56ad-4d53-9b24-58a56a70c5be","type":"ColumnDataSource"},"glyph":{"id":"9e09554c-d3e2-49b3-b66a-42a54585f55e","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"28ec02f0-91d7-4902-b2f2-4be710c72e74","type":"GlyphRenderer"},{"attributes":{"data_source":{"id":"84266960-5a18-4008-ae9e-75e28e9381f3","type":"ColumnDataSource"},"glyph":{"id":"1f6e0152-8147-41fe-b582-da7bfdf4eea7","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"d4645cb1-f94a-402c-8467-c4243899ef5f","type":"GlyphRenderer"},{"attributes":{"data_source":{"id":"c62cb59d-7624-4c33-b9d1-a60edbb8f71e","type":"ColumnDataSource"},"glyph":{"id":"0bbe18c7-2ac9-479b-a1a8-ed5bb982c779","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"23cc9f7b-226a-4964-8b16-f2be39bbf1a4","type":"GlyphRenderer"},{"attributes":{"data_source":{"id":"7e9b1595-b118-4aa2-a466-fd911dce7acd","type":"ColumnDataSource"},"glyph":{"id":"fbb1b3fc-2aaa-43e2-b649-b2e3c56523ff","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"4e486b41-62b5-480d-ba96-3dc6fb2f03c3","type":"GlyphRenderer"},{"attributes":{"axis_label":"Count( Fare )","formatter":{"id":"73115e39-86c0-4b48-ba94-f2901bd694d6","type":"BasicTickFormatter"},"plot":{"id":"69634d55-8bb5-4411-9dcb-fa5caf7293df","subtype":"Chart","type":"Plot"},"ticker":{"id":"209c695d-e3b3-402c-9a30-9c59bd343a6e","type":"BasicTicker"}},"id":"e0405173-ed14-4a66-9c93-1d826c6d8656","type":"LinearAxis"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(248.982415, 253.770538]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[0.0],"label":["(248.982415, 253.770538]"],"line_alpha":[1.0],"line_color":["black"],"width":[4.788123364485983],"x":["251.3764765"],"y":[0.0]}},"id":"842368d8-007a-4f10-91b1-b01ad63d393f","type":"ColumnDataSource"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"db32e410-fedc-4cdb-9cbf-66e462ea6526","type":"Rect"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"1b7c4cc4-f087-45e0-aa44-3dea32dae048","type":"Rect"},{"attributes":{},"id":"209c695d-e3b3-402c-9a30-9c59bd343a6e","type":"BasicTicker"},{"attributes":{"dimension":1,"plot":{"id":"69634d55-8bb5-4411-9dcb-fa5caf7293df","subtype":"Chart","type":"Plot"},"ticker":{"id":"209c695d-e3b3-402c-9a30-9c59bd343a6e","type":"BasicTicker"}},"id":"bc18e626-aa20-4b02-8c49-b0ac0100a925","type":"Grid"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(392.626116, 397.414239]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[0.0],"label":["(392.626116, 397.414239]"],"line_alpha":[1.0],"line_color":["black"],"width":[4.788123364486012],"x":["395.02017750000005"],"y":[0.0]}},"id":"bb9e298f-43dc-4b56-b0b0-d7916beae080","type":"ColumnDataSource"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"209a56d5-8b21-4c8a-90aa-450c372c5915","type":"Rect"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(397.414239, 402.202363]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[0.0],"label":["(397.414239, 402.202363]"],"line_alpha":[1.0],"line_color":["black"],"width":[4.788123364485955],"x":["399.80830100000003"],"y":[0.0]}},"id":"77e4e895-436a-4889-bdfe-06655b63e312","type":"ColumnDataSource"},{"attributes":{"data_source":{"id":"891ad0c3-67a6-4447-8025-f1aeb9f77329","type":"ColumnDataSource"},"glyph":{"id":"f3b1893c-7b8a-4a50-a0bc-0263adadd8c2","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"c2e7a6de-47d6-4821-a1fd-2347ecaa1e13","type":"GlyphRenderer"},{"attributes":{"callback":null,"end":344.3},"id":"619758aa-48e0-45c4-81b3-1cd2febac1d1","type":"Range1d"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(244.194292, 248.982415]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[2.0],"label":["(244.194292, 248.982415]"],"line_alpha":[1.0],"line_color":["black"],"width":[4.788123364485983],"x":["246.58835349999998"],"y":[1.0]}},"id":"4d837a3d-e4ce-4b0a-b7b8-ff7746366131","type":"ColumnDataSource"},{"attributes":{"data_source":{"id":"77e4e895-436a-4889-bdfe-06655b63e312","type":"ColumnDataSource"},"glyph":{"id":"db32e410-fedc-4cdb-9cbf-66e462ea6526","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"ea0fa54a-a8cf-4d6b-b142-370bcf9a0431","type":"GlyphRenderer"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(47.881234, 52.669357]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[18.0],"label":["(47.881234, 52.669357]"],"line_alpha":[1.0],"line_color":["black"],"width":[4.788123364485976],"x":["50.2752955"],"y":[9.0]}},"id":"9bd78e8d-0e88-4174-b001-95bd581f45ed","type":"ColumnDataSource"},{"attributes":{},"id":"b672df55-c50f-473f-ac30-060894585376","type":"ToolEvents"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(9.576247, 14.364370]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[109.0],"label":["(9.576247, 14.364370]"],"line_alpha":[1.0],"line_color":["black"],"width":[4.7881233644859815],"x":["11.9703085"],"y":[54.5]}},"id":"5c7417c0-15a2-4eb0-949c-3238d3519273","type":"ColumnDataSource"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"7546e39c-ce06-47fb-a33f-373e0b340557","type":"Rect"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(14.364370, 19.152493]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[69.0],"label":["(14.364370, 19.152493]"],"line_alpha":[1.0],"line_color":["black"],"width":[4.7881233644859815],"x":["16.7584315"],"y":[34.5]}},"id":"b54efd90-7daf-45e8-840d-a4b1ffa061af","type":"ColumnDataSource"},{"attributes":{"data_source":{"id":"5c7417c0-15a2-4eb0-949c-3238d3519273","type":"ColumnDataSource"},"glyph":{"id":"fcf8ea65-6fcc-4a2c-9f86-2a626136f16c","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"87835922-be3a-4e76-b86c-53bad896e9a4","type":"GlyphRenderer"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"4e39a552-1222-4b05-8ef2-6e9e47e9cbf2","type":"Rect"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(19.152493, 23.940617]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[40.0],"label":["(19.152493, 23.940617]"],"line_alpha":[1.0],"line_color":["black"],"width":[4.788123364485983],"x":["21.546554999999998"],"y":[20.0]}},"id":"f629dfbe-ec8e-4fca-b7a2-1c329a0ba80b","type":"ColumnDataSource"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"0409b86a-da84-408b-9621-7ca2121ae29f","type":"Rect"},{"attributes":{"data_source":{"id":"1d31b20d-f185-41e1-8f75-1cdfb09f2fb1","type":"ColumnDataSource"},"glyph":{"id":"836bf60c-6caf-4a86-b3c4-0a1be7786101","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"b2f4a511-15f1-4490-a2b4-b74cf898ebc9","type":"GlyphRenderer"},{"attributes":{"data_source":{"id":"b54efd90-7daf-45e8-840d-a4b1ffa061af","type":"ColumnDataSource"},"glyph":{"id":"7546e39c-ce06-47fb-a33f-373e0b340557","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"bc86ce32-7da3-4c3e-94e1-33ae5c84a207","type":"GlyphRenderer"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"eb55a817-3c8b-4f40-bfcd-e75bd9694fd6","type":"Rect"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(23.940617, 28.728740]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[94.0],"label":["(23.940617, 28.728740]"],"line_alpha":[1.0],"line_color":["black"],"width":[4.78812336448598],"x":["26.3346785"],"y":[47.0]}},"id":"b3debced-8bc7-495c-9ecb-f379a6881073","type":"ColumnDataSource"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"10c521ba-3433-4b50-ad85-a8accc972910","type":"Rect"},{"attributes":{"data_source":{"id":"f629dfbe-ec8e-4fca-b7a2-1c329a0ba80b","type":"ColumnDataSource"},"glyph":{"id":"4e39a552-1222-4b05-8ef2-6e9e47e9cbf2","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"d3907521-383c-485a-a7c7-9e3f22741d88","type":"GlyphRenderer"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"a0bf0ef9-128a-4a0e-888f-200d8995bd49","type":"Rect"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(28.728740, 33.516864]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[45.0],"label":["(28.728740, 33.516864]"],"line_alpha":[1.0],"line_color":["black"],"width":[4.78812336448598],"x":["31.122802"],"y":[22.5]}},"id":"fbd184f6-0459-43f4-a86f-1107f763eccd","type":"ColumnDataSource"},{"attributes":{"data_source":{"id":"b3debced-8bc7-495c-9ecb-f379a6881073","type":"ColumnDataSource"},"glyph":{"id":"eb55a817-3c8b-4f40-bfcd-e75bd9694fd6","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"c5c44cd8-2f6f-4fb2-a17f-8e0065a8b5d4","type":"GlyphRenderer"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(43.093110, 47.881234]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[7.0],"label":["(43.093110, 47.881234]"],"line_alpha":[1.0],"line_color":["black"],"width":[4.788123364485983],"x":["45.487172"],"y":[3.5]}},"id":"abbdf718-00ea-4de4-bf4a-3649b156375d","type":"ColumnDataSource"},{"attributes":{"data_source":{"id":"fbd184f6-0459-43f4-a86f-1107f763eccd","type":"ColumnDataSource"},"glyph":{"id":"a0bf0ef9-128a-4a0e-888f-200d8995bd49","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"49847ada-cf06-4fba-8d8b-aeef42c23e84","type":"GlyphRenderer"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"836bf60c-6caf-4a86-b3c4-0a1be7786101","type":"Rect"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"76c581bb-e7cb-4805-9271-7f1470f4e10c","type":"Rect"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(38.304987, 43.093110]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[19.0],"label":["(38.304987, 43.093110]"],"line_alpha":[1.0],"line_color":["black"],"width":[4.788123364485983],"x":["40.6990485"],"y":[9.5]}},"id":"1d31b20d-f185-41e1-8f75-1cdfb09f2fb1","type":"ColumnDataSource"},{"attributes":{"data_source":{"id":"abbdf718-00ea-4de4-bf4a-3649b156375d","type":"ColumnDataSource"},"glyph":{"id":"76c581bb-e7cb-4805-9271-7f1470f4e10c","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"734eb95d-1de1-43fb-aaf2-7490b69550ef","type":"GlyphRenderer"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"652c0cee-24fd-4701-9d74-03964137bbfa","type":"Rect"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(52.669357, 57.457480]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[21.0],"label":["(52.669357, 57.457480]"],"line_alpha":[1.0],"line_color":["black"],"width":[4.788123364485983],"x":["55.0634185"],"y":[10.5]}},"id":"2e584062-83a3-44b3-9429-4c586f5da172","type":"ColumnDataSource"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"16a2f064-4463-41ef-b3a5-1db2e3b5d36d","type":"Rect"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(138.855578, 143.643701]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[0.0],"label":["(138.855578, 143.643701]"],"line_alpha":[1.0],"line_color":["black"],"width":[4.788123364485983],"x":["141.2496395"],"y":[0.0]}},"id":"5fd22b2e-4649-4aa4-8215-6466a8b90765","type":"ColumnDataSource"},{"attributes":{"below":[{"id":"44885f48-29b5-4a4f-8863-1f592e481d3a","type":"LinearAxis"}],"css_classes":null,"left":[{"id":"e0405173-ed14-4a66-9c93-1d826c6d8656","type":"LinearAxis"}],"renderers":[{"id":"1dd7695f-d7ce-402b-81fd-6e6e96805b10","type":"BoxAnnotation"},{"id":"8e3c2fe4-78fc-4339-aeb9-ebe81010e4f1","type":"GlyphRenderer"},{"id":"38347de4-0cc2-443d-bcf7-5493e3e55656","type":"GlyphRenderer"},{"id":"87835922-be3a-4e76-b86c-53bad896e9a4","type":"GlyphRenderer"},{"id":"bc86ce32-7da3-4c3e-94e1-33ae5c84a207","type":"GlyphRenderer"},{"id":"d3907521-383c-485a-a7c7-9e3f22741d88","type":"GlyphRenderer"},{"id":"c5c44cd8-2f6f-4fb2-a17f-8e0065a8b5d4","type":"GlyphRenderer"},{"id":"49847ada-cf06-4fba-8d8b-aeef42c23e84","type":"GlyphRenderer"},{"id":"c4d894fd-8859-4264-a1f5-bbec3258ac2b","type":"GlyphRenderer"},{"id":"b2f4a511-15f1-4490-a2b4-b74cf898ebc9","type":"GlyphRenderer"},{"id":"734eb95d-1de1-43fb-aaf2-7490b69550ef","type":"GlyphRenderer"},{"id":"8797cfa3-ff43-4106-a085-19f3a90e4de6","type":"GlyphRenderer"},{"id":"cda1dddd-2a60-423e-82ad-73e5cf62870d","type":"GlyphRenderer"},{"id":"11da045f-aad8-4452-85c1-7c334fd74e51","type":"GlyphRenderer"},{"id":"e78f65c5-959b-4b03-98dc-14b3c4277af8","type":"GlyphRenderer"},{"id":"53eef5cf-7344-4995-8ebb-98628697877a","type":"GlyphRenderer"},{"id":"4c145668-bef9-4da6-9889-a86befd65883","type":"GlyphRenderer"},{"id":"d92c5e48-1fe4-4c48-b72f-aec4e1b1a2e1","type":"GlyphRenderer"},{"id":"bf6654ae-19d0-49c7-a385-2bac2af58f0c","type":"GlyphRenderer"},{"id":"3bd1adcc-b144-491e-86ad-a43dac055f12","type":"GlyphRenderer"},{"id":"148e8167-725a-4f01-8abb-312427139352","type":"GlyphRenderer"},{"id":"d9a75e48-f1dd-489d-b155-49073990b5db","type":"GlyphRenderer"},{"id":"e075e43d-edda-4397-ad28-73276ac80a1a","type":"GlyphRenderer"},{"id":"59377d88-de08-4f92-911c-b12b1452fb59","type":"GlyphRenderer"},{"id":"c2e1f75b-5d52-4c26-888b-35a2093ef4c1","type":"GlyphRenderer"},{"id":"9af8ea7d-0545-4fd2-a91d-70ac3a0bf20e","type":"GlyphRenderer"},{"id":"e02fa3ba-bf22-459e-80f9-baf2ac6241d7","type":"GlyphRenderer"},{"id":"0ed1fd86-8126-4724-ad50-2a6d4dc4bfa6","type":"GlyphRenderer"},{"id":"b2dfae73-1f11-4627-92dd-194b6e5e6299","type":"GlyphRenderer"},{"id":"17320b76-6206-4424-aba5-35bddc1c8782","type":"GlyphRenderer"},{"id":"6aaeedcd-cb94-4250-b56e-5d11b4d929ca","type":"GlyphRenderer"},{"id":"1df48c7a-0d93-4ef4-add2-9c14feb1b3b7","type":"GlyphRenderer"},{"id":"504a9cd4-8fd2-4d55-879b-3a01a4d5fa26","type":"GlyphRenderer"},{"id":"d269632c-8a99-4e7b-892e-261d38f85a7a","type":"GlyphRenderer"},{"id":"2c7b3d26-80d8-4d25-ba8f-6c8d35458635","type":"GlyphRenderer"},{"id":"2927c6c0-3b2c-4430-a511-76d9e6b4098b","type":"GlyphRenderer"},{"id":"7dc3a134-b76f-4404-b3fc-c9a854aa18cd","type":"GlyphRenderer"},{"id":"fcaaa70e-ff8e-4caa-afe2-13543dcc3cea","type":"GlyphRenderer"},{"id":"dad68869-da6d-4413-82c3-c83289e53429","type":"GlyphRenderer"},{"id":"65af37c5-f956-495b-bbb9-27ed0d386670","type":"GlyphRenderer"},{"id":"5b7a7975-3b07-44a5-a8ef-2aded03017b8","type":"GlyphRenderer"},{"id":"8cbff69a-867b-420e-adb8-8e0b0a290510","type":"GlyphRenderer"},{"id":"f05da917-b54e-4d65-a40b-1ed8a0f6e3f9","type":"GlyphRenderer"},{"id":"04610b76-6c3c-4f86-828a-58ee394e4b4b","type":"GlyphRenderer"},{"id":"fff21382-b33c-47f1-9146-e747551fdb87","type":"GlyphRenderer"},{"id":"4635ae54-db50-4100-9609-b79ca57ae1db","type":"GlyphRenderer"},{"id":"b954ada1-1227-4367-aa0b-178969c148eb","type":"GlyphRenderer"},{"id":"d4645cb1-f94a-402c-8467-c4243899ef5f","type":"GlyphRenderer"},{"id":"c2e7a6de-47d6-4821-a1fd-2347ecaa1e13","type":"GlyphRenderer"},{"id":"8e8ce2e9-147b-42cb-a8d9-3afa977eeba0","type":"GlyphRenderer"},{"id":"639f83d6-9299-4079-bb0a-354a1370a385","type":"GlyphRenderer"},{"id":"8201e287-879e-4494-b184-39598bad8cc5","type":"GlyphRenderer"},{"id":"e2c8e095-0c6a-4e77-a1ad-cc2d62c878fc","type":"GlyphRenderer"},{"id":"40479d52-4915-4cc9-9954-e616a644212e","type":"GlyphRenderer"},{"id":"43fce338-300b-41b3-aa3f-f8700cdfceb5","type":"GlyphRenderer"},{"id":"2263ce4c-54a2-4000-97cb-6930fbce93a0","type":"GlyphRenderer"},{"id":"2b184194-8d03-4b29-a04b-a037a824827e","type":"GlyphRenderer"},{"id":"f31dc873-fcf3-4f91-881e-e57b3902ba2d","type":"GlyphRenderer"},{"id":"9071ba82-aeff-4846-b1f2-5178fb231726","type":"GlyphRenderer"},{"id":"eba94e39-cc54-4000-b69e-b29571b6a099","type":"GlyphRenderer"},{"id":"9f1353ec-5465-4c62-8a14-5ccb38490a0a","type":"GlyphRenderer"},{"id":"28ec02f0-91d7-4902-b2f2-4be710c72e74","type":"GlyphRenderer"},{"id":"b3be644d-3769-4f15-aa91-140ded76fa97","type":"GlyphRenderer"},{"id":"6672a8b7-1e18-4704-ac66-e553decf4fca","type":"GlyphRenderer"},{"id":"866851ec-cc19-4b50-88fd-ac2e11ea7446","type":"GlyphRenderer"},{"id":"b9c0e6fd-3a0c-4135-bd0b-aa6791fb3d9e","type":"GlyphRenderer"},{"id":"0aa6688c-1662-428c-b81f-7e3866c93f2d","type":"GlyphRenderer"},{"id":"ab7d6c5c-5466-4c59-9062-ba8bada4d739","type":"GlyphRenderer"},{"id":"6bd482f2-9ed2-45c3-b9da-32419c7904ab","type":"GlyphRenderer"},{"id":"7c679c9e-e44d-489a-94d4-d8868ea41926","type":"GlyphRenderer"},{"id":"eb16452d-ee8d-4c47-b549-c78ecafefef2","type":"GlyphRenderer"},{"id":"09557cba-4c8b-4ab1-adf2-291ef947a5d0","type":"GlyphRenderer"},{"id":"3b9753aa-eee4-4931-a1c2-bbeb3dd66f01","type":"GlyphRenderer"},{"id":"d9f0ef5a-2ec3-4241-9aea-2589372cc9a6","type":"GlyphRenderer"},{"id":"e85896bc-96e5-43d2-a582-d0747aa282e0","type":"GlyphRenderer"},{"id":"a2338be1-3d49-4a56-9600-9b2db126c49b","type":"GlyphRenderer"},{"id":"22d42218-4eeb-4560-ba6f-fcb471b24c43","type":"GlyphRenderer"},{"id":"73fc26dd-106a-4562-acc7-3eb2dbb00906","type":"GlyphRenderer"},{"id":"224b2780-e10f-45e4-8ccf-f150523cdd45","type":"GlyphRenderer"},{"id":"3ca91b0b-4174-42fd-be45-77659c95ce53","type":"GlyphRenderer"},{"id":"03383d13-6de8-4d05-95f8-fc4afbb7e6fe","type":"GlyphRenderer"},{"id":"23cc9f7b-226a-4964-8b16-f2be39bbf1a4","type":"GlyphRenderer"},{"id":"3905cb26-8d9e-4248-a882-f0ad151f4fcf","type":"GlyphRenderer"},{"id":"7a10a4ea-f93f-4d28-ae3b-e610f065c21a","type":"GlyphRenderer"},{"id":"ea0fa54a-a8cf-4d6b-b142-370bcf9a0431","type":"GlyphRenderer"},{"id":"1bf9bc35-453d-4496-8ad1-fdd404121432","type":"GlyphRenderer"},{"id":"aa5603db-a522-41be-8cfb-aba46262deba","type":"GlyphRenderer"},{"id":"945d053a-318d-4908-8f5f-18ef9d87d5b5","type":"GlyphRenderer"},{"id":"5b8a07a6-3b7d-418d-81e2-22da3c6b7b52","type":"GlyphRenderer"},{"id":"a9a93c47-b850-4814-9567-408ecf509975","type":"GlyphRenderer"},{"id":"9dc5e624-f74c-44a4-9598-a764af6739bd","type":"GlyphRenderer"},{"id":"a5a12f98-cd1c-4727-8307-8ca55be9afc4","type":"GlyphRenderer"},{"id":"c7433078-1b9f-4c85-a2bd-9824873a2e5c","type":"GlyphRenderer"},{"id":"f70470bb-36fe-4605-9fe5-7e79ad02cffa","type":"GlyphRenderer"},{"id":"f95e19cb-5279-40c1-bf62-8fde6fd6aa73","type":"GlyphRenderer"},{"id":"1c1c76d1-a2c9-41f5-ae11-4b84b6d59d2d","type":"GlyphRenderer"},{"id":"d5e6b516-dfc4-41e9-aa19-7282e6ff848c","type":"GlyphRenderer"},{"id":"fcf0f7ba-6b6a-4c86-91f2-d8916f6cfdf0","type":"GlyphRenderer"},{"id":"41f503d9-58d5-43ef-b60b-326cff46bc16","type":"GlyphRenderer"},{"id":"c5e9547f-11d0-4f9d-88bc-a7a3b4337ff4","type":"GlyphRenderer"},{"id":"3a77f097-323a-4090-9bed-ea29b4e74493","type":"GlyphRenderer"},{"id":"5010eb1b-56b9-43d2-b867-27c54c642c21","type":"GlyphRenderer"},{"id":"b7143d08-3fbc-45d4-aa96-16f93885ee6c","type":"GlyphRenderer"},{"id":"dd3988f5-9902-4f87-ae09-1578ccfa9382","type":"GlyphRenderer"},{"id":"a8628e47-de93-4c7b-be4c-ad0189a0c472","type":"GlyphRenderer"},{"id":"cec476f8-4b39-48ac-802b-e7ea6451df31","type":"GlyphRenderer"},{"id":"8de9b562-8388-412b-bceb-a9678f96283f","type":"GlyphRenderer"},{"id":"4e486b41-62b5-480d-ba96-3dc6fb2f03c3","type":"GlyphRenderer"},{"id":"9f134643-c401-4357-9ab8-5430474df8bc","type":"Legend"},{"id":"44885f48-29b5-4a4f-8863-1f592e481d3a","type":"LinearAxis"},{"id":"e0405173-ed14-4a66-9c93-1d826c6d8656","type":"LinearAxis"},{"id":"bc18e626-aa20-4b02-8c49-b0ac0100a925","type":"Grid"}],"title":{"id":"2d2f9998-8631-4dae-bd16-8e451229583c","type":"Title"},"tool_events":{"id":"b672df55-c50f-473f-ac30-060894585376","type":"ToolEvents"},"toolbar":{"id":"7455eb31-3ca3-4082-be2e-c55b9b2e0cf9","type":"Toolbar"},"x_mapper_type":"auto","x_range":{"id":"328bece2-6d88-47b7-8faf-7e76ed9b563e","type":"Range1d"},"y_mapper_type":"auto","y_range":{"id":"619758aa-48e0-45c4-81b3-1cd2febac1d1","type":"Range1d"}},"id":"69634d55-8bb5-4411-9dcb-fa5caf7293df","subtype":"Chart","type":"Plot"},{"attributes":{"data_source":{"id":"9bd78e8d-0e88-4174-b001-95bd581f45ed","type":"ColumnDataSource"},"glyph":{"id":"0409b86a-da84-408b-9621-7ca2121ae29f","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"8797cfa3-ff43-4106-a085-19f3a90e4de6","type":"GlyphRenderer"},{"attributes":{"callback":null,"end":525.137430182243,"start":-12.808230182242992},"id":"328bece2-6d88-47b7-8faf-7e76ed9b563e","type":"Range1d"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(57.457480, 62.245604]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[6.0],"label":["(57.457480, 62.245604]"],"line_alpha":[1.0],"line_color":["black"],"width":[4.788123364485983],"x":["59.851541999999995"],"y":[3.0]}},"id":"5ba2267c-0564-4459-a56d-c7aa164b7a47","type":"ColumnDataSource"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"9079a1f5-fdc8-4870-b0f0-f22f077d0513","type":"Rect"},{"attributes":{"plot":null,"text":null},"id":"2d2f9998-8631-4dae-bd16-8e451229583c","type":"Title"},{"attributes":{"active_drag":"auto","active_scroll":"auto","active_tap":"auto","tools":[{"id":"bc22afce-b7a1-4dee-8b14-b07dd4159c09","type":"PanTool"},{"id":"e815ef17-dc07-40a7-99a5-4cc09d4a85c9","type":"WheelZoomTool"},{"id":"8774d805-25c6-4591-b268-12a4f6359b8b","type":"BoxZoomTool"},{"id":"62410d3e-61b8-4405-9f6a-04dc78706570","type":"SaveTool"},{"id":"27c4d39f-0576-438a-9a20-1e12b9a11372","type":"ResetTool"},{"id":"6f2a5974-5e99-44a8-90e1-b416fc63061e","type":"HelpTool"}]},"id":"7455eb31-3ca3-4082-be2e-c55b9b2e0cf9","type":"Toolbar"},{"attributes":{"data_source":{"id":"0a9a2219-14c7-41b8-8041-8525f315c787","type":"ColumnDataSource"},"glyph":{"id":"10c521ba-3433-4b50-ad85-a8accc972910","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"4c145668-bef9-4da6-9889-a86befd65883","type":"GlyphRenderer"},{"attributes":{"location":"top_left","plot":{"id":"69634d55-8bb5-4411-9dcb-fa5caf7293df","subtype":"Chart","type":"Plot"}},"id":"9f134643-c401-4357-9ab8-5430474df8bc","type":"Legend"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(105.338714, 110.126837]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[4.0],"label":["(105.338714, 110.126837]"],"line_alpha":[1.0],"line_color":["black"],"width":[4.788123364485983],"x":["107.7327755"],"y":[2.0]}},"id":"ef28c94f-4c01-4fea-9f8b-d0a1abdd8271","type":"ColumnDataSource"},{"attributes":{"data_source":{"id":"2e584062-83a3-44b3-9429-4c586f5da172","type":"ColumnDataSource"},"glyph":{"id":"652c0cee-24fd-4701-9d74-03964137bbfa","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"cda1dddd-2a60-423e-82ad-73e5cf62870d","type":"GlyphRenderer"},{"attributes":{"plot":{"id":"69634d55-8bb5-4411-9dcb-fa5caf7293df","subtype":"Chart","type":"Plot"}},"id":"bc22afce-b7a1-4dee-8b14-b07dd4159c09","type":"PanTool"},{"attributes":{"overlay":{"id":"1dd7695f-d7ce-402b-81fd-6e6e96805b10","type":"BoxAnnotation"},"plot":{"id":"69634d55-8bb5-4411-9dcb-fa5caf7293df","subtype":"Chart","type":"Plot"}},"id":"8774d805-25c6-4591-b268-12a4f6359b8b","type":"BoxZoomTool"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(71.821850, 76.609974]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[7.0],"label":["(71.821850, 76.609974]"],"line_alpha":[1.0],"line_color":["black"],"width":[4.788123364485983],"x":["74.215912"],"y":[3.5]}},"id":"0a9a2219-14c7-41b8-8041-8525f315c787","type":"ColumnDataSource"},{"attributes":{"plot":{"id":"69634d55-8bb5-4411-9dcb-fa5caf7293df","subtype":"Chart","type":"Plot"}},"id":"e815ef17-dc07-40a7-99a5-4cc09d4a85c9","type":"WheelZoomTool"},{"attributes":{"data_source":{"id":"5ba2267c-0564-4459-a56d-c7aa164b7a47","type":"ColumnDataSource"},"glyph":{"id":"9079a1f5-fdc8-4870-b0f0-f22f077d0513","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"11da045f-aad8-4452-85c1-7c334fd74e51","type":"GlyphRenderer"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"5d7842ba-ae27-49bc-a159-1916869b365b","type":"Rect"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(62.245604, 67.033727]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[5.0],"label":["(62.245604, 67.033727]"],"line_alpha":[1.0],"line_color":["black"],"width":[4.788123364485976],"x":["64.6396655"],"y":[2.5]}},"id":"ac9611e2-18ca-4a95-b030-5cd2a464d7c1","type":"ColumnDataSource"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(440.507350, 445.295473]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[0.0],"label":["(440.507350, 445.295473]"],"line_alpha":[1.0],"line_color":["black"],"width":[4.788123364486012],"x":["442.9014115"],"y":[0.0]}},"id":"c4372779-b378-4b59-8b28-61ba0690f9ee","type":"ColumnDataSource"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"d2208df4-57a3-4954-b230-91ff7b0bcc90","type":"Rect"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(296.863649, 301.651772]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[0.0],"label":["(296.863649, 301.651772]"],"line_alpha":[1.0],"line_color":["black"],"width":[4.788123364486012],"x":["299.25771050000003"],"y":[0.0]}},"id":"d4031349-363d-4e22-9c28-f1a79b3a2f74","type":"ColumnDataSource"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(402.202363, 406.990486]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[0.0],"label":["(402.202363, 406.990486]"],"line_alpha":[1.0],"line_color":["black"],"width":[4.788123364486012],"x":["404.5964245"],"y":[0.0]}},"id":"6dcfdf69-f4db-4cf1-a7a8-b6875e9b59b2","type":"ColumnDataSource"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"5ef19781-6724-45e3-8350-fdb98b4cbf30","type":"Rect"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"41c7fce8-d81e-4cee-a38c-150675110e05","type":"Rect"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(67.033727, 71.821850]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[12.0],"label":["(67.033727, 71.821850]"],"line_alpha":[1.0],"line_color":["black"],"width":[4.788123364485983],"x":["69.42778849999999"],"y":[6.0]}},"id":"4335768e-e772-4384-b51a-9672ea1c2f17","type":"ColumnDataSource"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(406.990486, 411.778609]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[0.0],"label":["(406.990486, 411.778609]"],"line_alpha":[1.0],"line_color":["black"],"width":[4.788123364485955],"x":["409.3845475"],"y":[0.0]}},"id":"52ad5cdf-14ce-4954-a638-af70da1566d8","type":"ColumnDataSource"},{"attributes":{"data_source":{"id":"6dcfdf69-f4db-4cf1-a7a8-b6875e9b59b2","type":"ColumnDataSource"},"glyph":{"id":"2c473fd7-0de2-4ed8-a5a2-edd6b5651bc7","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"1bf9bc35-453d-4496-8ad1-fdd404121432","type":"GlyphRenderer"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(292.075525, 296.863649]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[0.0],"label":["(292.075525, 296.863649]"],"line_alpha":[1.0],"line_color":["black"],"width":[4.788123364485955],"x":["294.46958700000005"],"y":[0.0]}},"id":"c1bf7e94-8459-479f-9d61-03fc93030b04","type":"ColumnDataSource"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"6bf67f31-2b9a-426d-aafe-a6122b735165","type":"Rect"},{"attributes":{"data_source":{"id":"c1bf7e94-8459-479f-9d61-03fc93030b04","type":"ColumnDataSource"},"glyph":{"id":"10b25519-3529-4c1b-9183-33291cece321","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"b3be644d-3769-4f15-aa91-140ded76fa97","type":"GlyphRenderer"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(411.778609, 416.566733]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[0.0],"label":["(411.778609, 416.566733]"],"line_alpha":[1.0],"line_color":["black"],"width":[4.788123364486012],"x":["414.17267100000004"],"y":[0.0]}},"id":"30f7f85b-f2e5-4686-ad14-f6de11d3b3ea","type":"ColumnDataSource"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(81.398097, 86.186221]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[8.0],"label":["(81.398097, 86.186221]"],"line_alpha":[1.0],"line_color":["black"],"width":[4.788123364485983],"x":["83.792159"],"y":[4.0]}},"id":"3ae49d42-f646-42f3-86a4-123146bd66c3","type":"ColumnDataSource"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"32fc5a00-95cb-4121-8c81-32817f7b4fd4","type":"Rect"},{"attributes":{"data_source":{"id":"ac9611e2-18ca-4a95-b030-5cd2a464d7c1","type":"ColumnDataSource"},"glyph":{"id":"d2208df4-57a3-4954-b230-91ff7b0bcc90","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"e78f65c5-959b-4b03-98dc-14b3c4277af8","type":"GlyphRenderer"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(301.651772, 306.439895]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[0.0],"label":["(301.651772, 306.439895]"],"line_alpha":[1.0],"line_color":["black"],"width":[4.788123364485955],"x":["304.04583349999996"],"y":[0.0]}},"id":"e891afa9-1946-4b34-a9ce-ccf1e4f6c2f7","type":"ColumnDataSource"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(76.609974, 81.398097]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[21.0],"label":["(76.609974, 81.398097]"],"line_alpha":[1.0],"line_color":["black"],"width":[4.788123364485983],"x":["79.0040355"],"y":[10.5]}},"id":"45938f58-126e-41fb-823b-5462776e2ecd","type":"ColumnDataSource"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"6903ac98-78ef-4ee3-b0e0-d0d9248a16f2","type":"Rect"},{"attributes":{"data_source":{"id":"84067b8d-0c2d-489b-a39f-3cea6a9896b6","type":"ColumnDataSource"},"glyph":{"id":"9963b008-561f-4d89-886c-375363166b21","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"a5a12f98-cd1c-4727-8307-8ca55be9afc4","type":"GlyphRenderer"},{"attributes":{"data_source":{"id":"52ad5cdf-14ce-4954-a638-af70da1566d8","type":"ColumnDataSource"},"glyph":{"id":"41c7fce8-d81e-4cee-a38c-150675110e05","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"aa5603db-a522-41be-8cfb-aba46262deba","type":"GlyphRenderer"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(330.380512, 335.168636]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[0.0],"label":["(330.380512, 335.168636]"],"line_alpha":[1.0],"line_color":["black"],"width":[4.788123364486012],"x":["332.77457400000003"],"y":[0.0]}},"id":"839d6ef5-46c1-4f2b-8105-d7d1c36d8f53","type":"ColumnDataSource"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"09da3028-6f26-4d82-a640-202103d982bb","type":"Rect"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"7bf4dccd-6b1c-4481-9a19-9adac28f1a12","type":"Rect"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(416.566733, 421.354856]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[0.0],"label":["(416.566733, 421.354856]"],"line_alpha":[1.0],"line_color":["black"],"width":[4.788123364485955],"x":["418.9607945"],"y":[0.0]}},"id":"0c0f7d2a-6bf3-4524-b0a4-f95edb7c5679","type":"ColumnDataSource"},{"attributes":{"data_source":{"id":"d4031349-363d-4e22-9c28-f1a79b3a2f74","type":"ColumnDataSource"},"glyph":{"id":"5d7842ba-ae27-49bc-a159-1916869b365b","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"6672a8b7-1e18-4704-ac66-e553decf4fca","type":"GlyphRenderer"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"b400e84a-8ebf-43c9-b6d1-fbb52319e04d","type":"Rect"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"9982178d-c4ae-4369-a3d6-a15702bb6442","type":"Rect"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"f32bf360-6a18-4aac-a222-e4f7d768ac24","type":"Rect"},{"attributes":{"data_source":{"id":"4335768e-e772-4384-b51a-9672ea1c2f17","type":"ColumnDataSource"},"glyph":{"id":"5ef19781-6724-45e3-8350-fdb98b4cbf30","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"53eef5cf-7344-4995-8ebb-98628697877a","type":"GlyphRenderer"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(306.439895, 311.228019]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[0.0],"label":["(306.439895, 311.228019]"],"line_alpha":[1.0],"line_color":["black"],"width":[4.788123364485955],"x":["308.833957"],"y":[0.0]}},"id":"dfd4ab9f-2974-475f-b524-90851c09e675","type":"ColumnDataSource"},{"attributes":{"data_source":{"id":"5fd22b2e-4649-4aa4-8215-6466a8b90765","type":"ColumnDataSource"},"glyph":{"id":"87457fac-a314-4ffa-bea4-a4c3d64069e9","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"6aaeedcd-cb94-4250-b56e-5d11b4d929ca","type":"GlyphRenderer"},{"attributes":{"data_source":{"id":"45938f58-126e-41fb-823b-5462776e2ecd","type":"ColumnDataSource"},"glyph":{"id":"16a2f064-4463-41ef-b3a5-1db2e3b5d36d","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"d92c5e48-1fe4-4c48-b72f-aec4e1b1a2e1","type":"GlyphRenderer"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(430.931103, 435.719226]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[0.0],"label":["(430.931103, 435.719226]"],"line_alpha":[1.0],"line_color":["black"],"width":[4.788123364486012],"x":["433.3251645"],"y":[0.0]}},"id":"84067b8d-0c2d-489b-a39f-3cea6a9896b6","type":"ColumnDataSource"},{"attributes":{"data_source":{"id":"5a6ba3c8-cb43-4036-903d-7e699b154749","type":"ColumnDataSource"},"glyph":{"id":"fab5a822-9245-4c03-82a6-d5f9686af8f0","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"a8628e47-de93-4c7b-be4c-ad0189a0c472","type":"GlyphRenderer"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"80a38598-b6f6-42c6-9bc3-1f581bd25474","type":"Rect"},{"attributes":{"data_source":{"id":"6f32e4fd-3944-4cd1-9ac2-ce2b23377cb4","type":"ColumnDataSource"},"glyph":{"id":"087dee78-0938-40b6-b74b-d1e0c74a2f25","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"6bd482f2-9ed2-45c3-b9da-32419c7904ab","type":"GlyphRenderer"},{"attributes":{"data_source":{"id":"30f7f85b-f2e5-4686-ad14-f6de11d3b3ea","type":"ColumnDataSource"},"glyph":{"id":"6bf67f31-2b9a-426d-aafe-a6122b735165","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"945d053a-318d-4908-8f5f-18ef9d87d5b5","type":"GlyphRenderer"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(86.186221, 90.974344]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[9.0],"label":["(86.186221, 90.974344]"],"line_alpha":[1.0],"line_color":["black"],"width":[4.788123364485983],"x":["88.58028250000001"],"y":[4.5]}},"id":"d8ae3389-8fe3-4bd1-a465-a0198f88ff04","type":"ColumnDataSource"},{"attributes":{"data_source":{"id":"e891afa9-1946-4b34-a9ce-ccf1e4f6c2f7","type":"ColumnDataSource"},"glyph":{"id":"32fc5a00-95cb-4121-8c81-32817f7b4fd4","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"866851ec-cc19-4b50-88fd-ac2e11ea7446","type":"GlyphRenderer"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(493.176707, 497.964830]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[0.0],"label":["(493.176707, 497.964830]"],"line_alpha":[1.0],"line_color":["black"],"width":[4.788123364486012],"x":["495.5707685"],"y":[0.0]}},"id":"5a6ba3c8-cb43-4036-903d-7e699b154749","type":"ColumnDataSource"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"e1e32a36-498a-4e8d-8f15-4000251b430b","type":"Rect"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(311.228019, 316.016142]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[0.0],"label":["(311.228019, 316.016142]"],"line_alpha":[1.0],"line_color":["black"],"width":[4.788123364486012],"x":["313.62208050000004"],"y":[0.0]}},"id":"ff9d25c7-527e-455a-94b0-6cfdf228b53a","type":"ColumnDataSource"},{"attributes":{"data_source":{"id":"3ae49d42-f646-42f3-86a4-123146bd66c3","type":"ColumnDataSource"},"glyph":{"id":"b400e84a-8ebf-43c9-b6d1-fbb52319e04d","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"bf6654ae-19d0-49c7-a385-2bac2af58f0c","type":"GlyphRenderer"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"fe5a5d68-4d75-488b-a5cb-5c1c15122af1","type":"Rect"},{"attributes":{"data_source":{"id":"0c0f7d2a-6bf3-4524-b0a4-f95edb7c5679","type":"ColumnDataSource"},"glyph":{"id":"7bf4dccd-6b1c-4481-9a19-9adac28f1a12","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"5b8a07a6-3b7d-418d-81e2-22da3c6b7b52","type":"GlyphRenderer"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(90.974344, 95.762467]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[4.0],"label":["(90.974344, 95.762467]"],"line_alpha":[1.0],"line_color":["black"],"width":[4.788123364485983],"x":["93.3684055"],"y":[2.0]}},"id":"394b8d49-b574-49f7-97bb-32b99d030748","type":"ColumnDataSource"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(320.804265, 325.592389]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[0.0],"label":["(320.804265, 325.592389]"],"line_alpha":[1.0],"line_color":["black"],"width":[4.788123364486012],"x":["323.198327"],"y":[0.0]}},"id":"6f32e4fd-3944-4cd1-9ac2-ce2b23377cb4","type":"ColumnDataSource"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"da12c5e6-4282-4d56-9e00-d8f5b0af577f","type":"Rect"},{"attributes":{"data_source":{"id":"dfd4ab9f-2974-475f-b524-90851c09e675","type":"ColumnDataSource"},"glyph":{"id":"9982178d-c4ae-4369-a3d6-a15702bb6442","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"b9c0e6fd-3a0c-4135-bd0b-aa6791fb3d9e","type":"GlyphRenderer"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(435.719226, 440.507350]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[0.0],"label":["(435.719226, 440.507350]"],"line_alpha":[1.0],"line_color":["black"],"width":[4.788123364485955],"x":["438.113288"],"y":[0.0]}},"id":"4ec6f683-99b7-4c7d-8be9-53e64a7d5ad4","type":"ColumnDataSource"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"a557d712-9dbc-4f5f-ac89-bfcf56bc4e68","type":"Rect"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"a1bb878c-be38-4795-b906-e574a87d415b","type":"Rect"},{"attributes":{"data_source":{"id":"d8ae3389-8fe3-4bd1-a465-a0198f88ff04","type":"ColumnDataSource"},"glyph":{"id":"80a38598-b6f6-42c6-9bc3-1f581bd25474","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"3bd1adcc-b144-491e-86ad-a43dac055f12","type":"GlyphRenderer"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"4fe3ce33-0af4-4345-b1a9-dc40dd82a99c","type":"Rect"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["[0.000000, 4.788123]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[16.0],"label":["[0.000000, 4.788123]"],"line_alpha":[1.0],"line_color":["black"],"width":[4.7881233644859815],"x":["2.3940615"],"y":[8.0]}},"id":"911713b9-186d-4368-b67e-e104ec069612","type":"ColumnDataSource"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(95.762467, 100.550591]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[0.0],"label":["(95.762467, 100.550591]"],"line_alpha":[1.0],"line_color":["black"],"width":[4.788123364485969],"x":["98.156529"],"y":[0.0]}},"id":"aba72d39-62a2-475d-8a79-28192c350bfe","type":"ColumnDataSource"},{"attributes":{"data_source":{"id":"edef4ed4-95ba-4b1a-887e-aee038f08b46","type":"ColumnDataSource"},"glyph":{"id":"89d570f5-10b1-4688-9f07-60b0072d7b2e","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"fcf0f7ba-6b6a-4c86-91f2-d8916f6cfdf0","type":"GlyphRenderer"},{"attributes":{"data_source":{"id":"ff9d25c7-527e-455a-94b0-6cfdf228b53a","type":"ColumnDataSource"},"glyph":{"id":"e1e32a36-498a-4e8d-8f15-4000251b430b","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"0aa6688c-1662-428c-b81f-7e3866c93f2d","type":"GlyphRenderer"},{"attributes":{"data_source":{"id":"4ec6f683-99b7-4c7d-8be9-53e64a7d5ad4","type":"ColumnDataSource"},"glyph":{"id":"da12c5e6-4282-4d56-9e00-d8f5b0af577f","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"c7433078-1b9f-4c85-a2bd-9824873a2e5c","type":"GlyphRenderer"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"2429c684-91d4-4cc1-ba87-1ce14b2be1c2","type":"Rect"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(445.295473, 450.083596]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[0.0],"label":["(445.295473, 450.083596]"],"line_alpha":[1.0],"line_color":["black"],"width":[4.788123364485955],"x":["447.68953450000004"],"y":[0.0]}},"id":"e2620d4a-9056-4493-a75a-e1c322eb0344","type":"ColumnDataSource"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"87457fac-a314-4ffa-bea4-a4c3d64069e9","type":"Rect"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(325.592389, 330.380512]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[0.0],"label":["(325.592389, 330.380512]"],"line_alpha":[1.0],"line_color":["black"],"width":[4.788123364485955],"x":["327.98645050000005"],"y":[0.0]}},"id":"394de5d7-b841-42c7-b83b-7d2ec1707844","type":"ColumnDataSource"},{"attributes":{"data_source":{"id":"54532b7f-0506-4aa0-b5e4-80b1de08ec0f","type":"ColumnDataSource"},"glyph":{"id":"c59cc036-ea66-4666-812b-3d5c55132729","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"c2e1f75b-5d52-4c26-888b-35a2093ef4c1","type":"GlyphRenderer"},{"attributes":{"data_source":{"id":"394b8d49-b574-49f7-97bb-32b99d030748","type":"ColumnDataSource"},"glyph":{"id":"fe5a5d68-4d75-488b-a5cb-5c1c15122af1","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"148e8167-725a-4f01-8abb-312427139352","type":"GlyphRenderer"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"530c0348-57fa-4b9c-b870-67b9fd345b48","type":"Rect"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"087dee78-0938-40b6-b74b-d1e0c74a2f25","type":"Rect"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(100.550591, 105.338714]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[0.0],"label":["(100.550591, 105.338714]"],"line_alpha":[1.0],"line_color":["black"],"width":[4.788123364485983],"x":["102.94465249999999"],"y":[0.0]}},"id":"b51fdb84-19ad-493c-a551-a00720459390","type":"ColumnDataSource"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"9963b008-561f-4d89-886c-375363166b21","type":"Rect"},{"attributes":{"data_source":{"id":"c4372779-b378-4b59-8b28-61ba0690f9ee","type":"ColumnDataSource"},"glyph":{"id":"6903ac98-78ef-4ee3-b0e0-d0d9248a16f2","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"f70470bb-36fe-4605-9fe5-7e79ad02cffa","type":"GlyphRenderer"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"6c10ad25-5f59-4f68-9289-a4ac7c186cad","type":"Rect"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"9f0e53d1-4c22-4641-ac6f-203bfad85e15","type":"Rect"},{"attributes":{"data_source":{"id":"394de5d7-b841-42c7-b83b-7d2ec1707844","type":"ColumnDataSource"},"glyph":{"id":"a557d712-9dbc-4f5f-ac89-bfcf56bc4e68","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"7c679c9e-e44d-489a-94d4-d8868ea41926","type":"GlyphRenderer"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"3429f133-dcbf-41a4-ae1a-ba7d92b9127e","type":"Rect"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(450.083596, 454.871720]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[0.0],"label":["(450.083596, 454.871720]"],"line_alpha":[1.0],"line_color":["black"],"width":[4.788123364486012],"x":["452.477658"],"y":[0.0]}},"id":"c39dd26d-4306-43b1-8bc9-16d9eaf382bb","type":"ColumnDataSource"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"ad279b9a-2b66-4ed3-b721-01f3eb29d3a3","type":"Rect"},{"attributes":{"data_source":{"id":"aba72d39-62a2-475d-8a79-28192c350bfe","type":"ColumnDataSource"},"glyph":{"id":"4fe3ce33-0af4-4345-b1a9-dc40dd82a99c","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"d9a75e48-f1dd-489d-b155-49073990b5db","type":"GlyphRenderer"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(335.168636, 339.956759]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[0.0],"label":["(335.168636, 339.956759]"],"line_alpha":[1.0],"line_color":["black"],"width":[4.788123364485955],"x":["337.5626975"],"y":[0.0]}},"id":"6fb53363-478e-47a9-9c49-a44b67d37065","type":"ColumnDataSource"},{"attributes":{"data_source":{"id":"ef28c94f-4c01-4fea-9f8b-d0a1abdd8271","type":"ColumnDataSource"},"glyph":{"id":"d7dd59f9-0b95-46b1-bfa9-c4b6dd1e43e2","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"59377d88-de08-4f92-911c-b12b1452fb59","type":"GlyphRenderer"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"c59cc036-ea66-4666-812b-3d5c55132729","type":"Rect"},{"attributes":{"data_source":{"id":"7d7e35ed-98d9-4019-984b-f5ca653a10b2","type":"ColumnDataSource"},"glyph":{"id":"48bf0fe6-4875-422e-bf56-3397e1314ebf","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"a2338be1-3d49-4a56-9600-9b2db126c49b","type":"GlyphRenderer"},{"attributes":{"data_source":{"id":"e2620d4a-9056-4493-a75a-e1c322eb0344","type":"ColumnDataSource"},"glyph":{"id":"2429c684-91d4-4cc1-ba87-1ce14b2be1c2","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"f95e19cb-5279-40c1-bf62-8fde6fd6aa73","type":"GlyphRenderer"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"b0a76730-f65a-4845-9e2b-ffbc7b8bab48","type":"Rect"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"884630e1-e820-4e8d-a210-d3ed7bf8e12b","type":"Rect"},{"attributes":{"data_source":{"id":"839d6ef5-46c1-4f2b-8105-d7d1c36d8f53","type":"ColumnDataSource"},"glyph":{"id":"09da3028-6f26-4d82-a640-202103d982bb","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"eb16452d-ee8d-4c47-b549-c78ecafefef2","type":"GlyphRenderer"},{"attributes":{"data_source":{"id":"d36cb02e-0114-40d9-8d6e-31128de92dbf","type":"ColumnDataSource"},"glyph":{"id":"b0a76730-f65a-4845-9e2b-ffbc7b8bab48","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"d5e6b516-dfc4-41e9-aa19-7282e6ff848c","type":"GlyphRenderer"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(339.956759, 344.744882]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[0.0],"label":["(339.956759, 344.744882]"],"line_alpha":[1.0],"line_color":["black"],"width":[4.788123364486012],"x":["342.3508205"],"y":[0.0]}},"id":"4988fc2d-c2fa-4a0a-84c4-5ca797ba6fad","type":"ColumnDataSource"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"90278d7b-f794-4824-84ce-703924e7bc1b","type":"Rect"},{"attributes":{"data_source":{"id":"b51fdb84-19ad-493c-a551-a00720459390","type":"ColumnDataSource"},"glyph":{"id":"530c0348-57fa-4b9c-b870-67b9fd345b48","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"e075e43d-edda-4397-ad28-73276ac80a1a","type":"GlyphRenderer"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"27cf795e-ecf0-4be6-839a-3c8f7d5a3594","type":"Rect"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"89d570f5-10b1-4688-9f07-60b0072d7b2e","type":"Rect"},{"attributes":{"data_source":{"id":"c39dd26d-4306-43b1-8bc9-16d9eaf382bb","type":"ColumnDataSource"},"glyph":{"id":"9f0e53d1-4c22-4641-ac6f-203bfad85e15","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"1c1c76d1-a2c9-41f5-ae11-4b84b6d59d2d","type":"GlyphRenderer"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(110.126837, 114.914961]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[7.0],"label":["(110.126837, 114.914961]"],"line_alpha":[1.0],"line_color":["black"],"width":[4.788123364485983],"x":["112.520899"],"y":[3.5]}},"id":"54532b7f-0506-4aa0-b5e4-80b1de08ec0f","type":"ColumnDataSource"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(387.837993, 392.626116]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[0.0],"label":["(387.837993, 392.626116]"],"line_alpha":[1.0],"line_color":["black"],"width":[4.788123364485955],"x":["390.2320545"],"y":[0.0]}},"id":"8f76e89c-9442-4dbe-ae5e-a02b0125fc97","type":"ColumnDataSource"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(459.659843, 464.447966]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[0.0],"label":["(459.659843, 464.447966]"],"line_alpha":[1.0],"line_color":["black"],"width":[4.788123364485955],"x":["462.05390450000004"],"y":[0.0]}},"id":"edef4ed4-95ba-4b1a-887e-aee038f08b46","type":"ColumnDataSource"},{"attributes":{"data_source":{"id":"6fb53363-478e-47a9-9c49-a44b67d37065","type":"ColumnDataSource"},"glyph":{"id":"ad279b9a-2b66-4ed3-b721-01f3eb29d3a3","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"09557cba-4c8b-4ab1-adf2-291ef947a5d0","type":"GlyphRenderer"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"d7dd59f9-0b95-46b1-bfa9-c4b6dd1e43e2","type":"Rect"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(114.914961, 119.703084]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[0.0],"label":["(114.914961, 119.703084]"],"line_alpha":[1.0],"line_color":["black"],"width":[4.788123364485983],"x":["117.3090225"],"y":[0.0]}},"id":"7406fb4c-a384-4653-9e0a-d844c1ebc584","type":"ColumnDataSource"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(454.871720, 459.659843]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[0.0],"label":["(454.871720, 459.659843]"],"line_alpha":[1.0],"line_color":["black"],"width":[4.788123364485955],"x":["457.2657815"],"y":[0.0]}},"id":"d36cb02e-0114-40d9-8d6e-31128de92dbf","type":"ColumnDataSource"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"fab5a822-9245-4c03-82a6-d5f9686af8f0","type":"Rect"},{"attributes":{"data_source":{"id":"7406fb4c-a384-4653-9e0a-d844c1ebc584","type":"ColumnDataSource"},"glyph":{"id":"27cf795e-ecf0-4be6-839a-3c8f7d5a3594","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"9af8ea7d-0545-4fd2-a91d-70ac3a0bf20e","type":"GlyphRenderer"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(354.321129, 359.109252]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[0.0],"label":["(354.321129, 359.109252]"],"line_alpha":[1.0],"line_color":["black"],"width":[4.788123364485955],"x":["356.7151905"],"y":[0.0]}},"id":"7d7e35ed-98d9-4019-984b-f5ca653a10b2","type":"ColumnDataSource"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"c29aaeb6-7317-4d29-acab-785ea31d1deb","type":"Rect"},{"attributes":{"plot":{"id":"69634d55-8bb5-4411-9dcb-fa5caf7293df","subtype":"Chart","type":"Plot"}},"id":"62410d3e-61b8-4405-9f6a-04dc78706570","type":"SaveTool"},{"attributes":{"bottom_units":"screen","fill_alpha":{"value":0.5},"fill_color":{"value":"lightgrey"},"left_units":"screen","level":"overlay","line_alpha":{"value":1.0},"line_color":{"value":"black"},"line_dash":[4,4],"line_width":{"value":2},"plot":null,"render_mode":"css","right_units":"screen","top_units":"screen"},"id":"1dd7695f-d7ce-402b-81fd-6e6e96805b10","type":"BoxAnnotation"},{"attributes":{"plot":{"id":"69634d55-8bb5-4411-9dcb-fa5caf7293df","subtype":"Chart","type":"Plot"}},"id":"27c4d39f-0576-438a-9a20-1e12b9a11372","type":"ResetTool"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(234.618045, 239.406168]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[0.0],"label":["(234.618045, 239.406168]"],"line_alpha":[1.0],"line_color":["black"],"width":[4.788123364485983],"x":["237.01210650000002"],"y":[0.0]}},"id":"0be7e9f9-3b90-4db8-acb0-08da56cdec27","type":"ColumnDataSource"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(229.829921, 234.618045]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[0.0],"label":["(229.829921, 234.618045]"],"line_alpha":[1.0],"line_color":["black"],"width":[4.788123364485983],"x":["232.223983"],"y":[0.0]}},"id":"cf86cfa9-12e3-410c-886e-160f2ba553b7","type":"ColumnDataSource"},{"attributes":{"plot":{"id":"69634d55-8bb5-4411-9dcb-fa5caf7293df","subtype":"Chart","type":"Plot"}},"id":"6f2a5974-5e99-44a8-90e1-b416fc63061e","type":"HelpTool"},{"attributes":{"data_source":{"id":"cf86cfa9-12e3-410c-886e-160f2ba553b7","type":"ColumnDataSource"},"glyph":{"id":"1b7c4cc4-f087-45e0-aa44-3dea32dae048","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"8e8ce2e9-147b-42cb-a8d9-3afa977eeba0","type":"GlyphRenderer"},{"attributes":{},"id":"73115e39-86c0-4b48-ba94-f2901bd694d6","type":"BasicTickFormatter"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"b40077da-510b-4245-961c-05462b23d3c0","type":"Rect"},{"attributes":{},"id":"8f72211a-b696-4037-b0dd-67d86ef9ef1c","type":"BasicTickFormatter"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(239.406168, 244.194292]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[0.0],"label":["(239.406168, 244.194292]"],"line_alpha":[1.0],"line_color":["black"],"width":[4.788123364485983],"x":["241.80023"],"y":[0.0]}},"id":"f93cc1f4-89cb-488c-9278-9cf4d50bd8e1","type":"ColumnDataSource"},{"attributes":{},"id":"74154beb-0beb-4edc-901c-bf7156feae79","type":"BasicTicker"},{"attributes":{"data_source":{"id":"0be7e9f9-3b90-4db8-acb0-08da56cdec27","type":"ColumnDataSource"},"glyph":{"id":"c29aaeb6-7317-4d29-acab-785ea31d1deb","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"639f83d6-9299-4079-bb0a-354a1370a385","type":"GlyphRenderer"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"c182cba8-eca8-4016-9462-967304799f19","type":"Rect"},{"attributes":{"data_source":{"id":"842368d8-007a-4f10-91b1-b01ad63d393f","type":"ColumnDataSource"},"glyph":{"id":"afdf05a7-c7e0-4cc5-9465-a49e1f660f83","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"40479d52-4915-4cc9-9954-e616a644212e","type":"GlyphRenderer"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(258.558662, 263.346785]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[6.0],"label":["(258.558662, 263.346785]"],"line_alpha":[1.0],"line_color":["black"],"width":[4.788123364486012],"x":["260.95272350000005"],"y":[3.0]}},"id":"09d99695-5e27-4202-974e-d9723e466d8e","type":"ColumnDataSource"},{"attributes":{"data_source":{"id":"f93cc1f4-89cb-488c-9278-9cf4d50bd8e1","type":"ColumnDataSource"},"glyph":{"id":"b40077da-510b-4245-961c-05462b23d3c0","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"8201e287-879e-4494-b184-39598bad8cc5","type":"GlyphRenderer"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(253.770538, 258.558662]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[0.0],"label":["(253.770538, 258.558662]"],"line_alpha":[1.0],"line_color":["black"],"width":[4.788123364485955],"x":["256.1646"],"y":[0.0]}},"id":"df64c780-1695-49e3-b722-a495aae4e555","type":"ColumnDataSource"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"6e61188a-1f8b-445b-b05c-73dd7c817d1b","type":"Rect"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"48bf0fe6-4875-422e-bf56-3397e1314ebf","type":"Rect"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"7a533c2e-574e-41b6-b973-8b076e30670e","type":"Rect"},{"attributes":{"data_source":{"id":"4885855e-1951-402e-b83f-63eddcc58a22","type":"ColumnDataSource"},"glyph":{"id":"f32bf360-6a18-4aac-a222-e4f7d768ac24","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"9dc5e624-f74c-44a4-9598-a764af6739bd","type":"GlyphRenderer"},{"attributes":{"data_source":{"id":"4d837a3d-e4ce-4b0a-b7b8-ff7746366131","type":"ColumnDataSource"},"glyph":{"id":"c182cba8-eca8-4016-9462-967304799f19","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"e2c8e095-0c6a-4e77-a1ad-cc2d62c878fc","type":"GlyphRenderer"},{"attributes":{"data_source":{"id":"df64c780-1695-49e3-b722-a495aae4e555","type":"ColumnDataSource"},"glyph":{"id":"6e61188a-1f8b-445b-b05c-73dd7c817d1b","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"43fce338-300b-41b3-aa3f-f8700cdfceb5","type":"GlyphRenderer"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"01edf3fd-c1a3-42e6-98d3-bb1229bd1b2c","type":"Rect"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(263.346785, 268.134908]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[0.0],"label":["(263.346785, 268.134908]"],"line_alpha":[1.0],"line_color":["black"],"width":[4.788123364485955],"x":["265.7408465"],"y":[0.0]}},"id":"eb8e3d52-1198-4748-84a4-f05372ca69b9","type":"ColumnDataSource"},{"attributes":{"data_source":{"id":"09d99695-5e27-4202-974e-d9723e466d8e","type":"ColumnDataSource"},"glyph":{"id":"7a533c2e-574e-41b6-b973-8b076e30670e","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"2263ce4c-54a2-4000-97cb-6930fbce93a0","type":"GlyphRenderer"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"278241ce-818e-48bc-ba47-0adde30ba66f","type":"Rect"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(268.134908, 272.923032]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[0.0],"label":["(268.134908, 272.923032]"],"line_alpha":[1.0],"line_color":["black"],"width":[4.788123364486012],"x":["270.52896999999996"],"y":[0.0]}},"id":"2bfcfd92-fb4b-42ee-af43-ea7f8038e486","type":"ColumnDataSource"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(426.142979, 430.931103]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[0.0],"label":["(426.142979, 430.931103]"],"line_alpha":[1.0],"line_color":["black"],"width":[4.788123364485955],"x":["428.53704100000004"],"y":[0.0]}},"id":"4885855e-1951-402e-b83f-63eddcc58a22","type":"ColumnDataSource"},{"attributes":{"data_source":{"id":"eb8e3d52-1198-4748-84a4-f05372ca69b9","type":"ColumnDataSource"},"glyph":{"id":"01edf3fd-c1a3-42e6-98d3-bb1229bd1b2c","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"2b184194-8d03-4b29-a04b-a037a824827e","type":"GlyphRenderer"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"1141ac64-e908-452d-8ac6-16f4dbda59a6","type":"Rect"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(272.923032, 277.711155]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[0.0],"label":["(272.923032, 277.711155]"],"line_alpha":[1.0],"line_color":["black"],"width":[4.788123364485955],"x":["275.3170935"],"y":[0.0]}},"id":"c72d7375-5912-4bad-9c19-77befe97d3cc","type":"ColumnDataSource"},{"attributes":{"data_source":{"id":"7c26cf5a-a7cc-4a90-a470-f0b42d15b5b0","type":"ColumnDataSource"},"glyph":{"id":"0f1abf8f-a366-4eac-86d1-01e0288447ad","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"9f1353ec-5465-4c62-8a14-5ccb38490a0a","type":"GlyphRenderer"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"9e09554c-d3e2-49b3-b66a-42a54585f55e","type":"Rect"},{"attributes":{"data_source":{"id":"2bfcfd92-fb4b-42ee-af43-ea7f8038e486","type":"ColumnDataSource"},"glyph":{"id":"278241ce-818e-48bc-ba47-0adde30ba66f","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"f31dc873-fcf3-4f91-881e-e57b3902ba2d","type":"GlyphRenderer"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(33.516864, 38.304987]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[15.0],"label":["(33.516864, 38.304987]"],"line_alpha":[1.0],"line_color":["black"],"width":[4.788123364485983],"x":["35.9109255"],"y":[7.5]}},"id":"859b1702-a075-40c7-ae7e-3eb5a5ea504f","type":"ColumnDataSource"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"eb4ee46f-0ccd-4fcb-a23d-c3fa7cdc8f0c","type":"Rect"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(277.711155, 282.499279]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[0.0],"label":["(277.711155, 282.499279]"],"line_alpha":[1.0],"line_color":["black"],"width":[4.788123364486012],"x":["280.10521700000004"],"y":[0.0]}},"id":"d69a780b-973d-45aa-9fde-bf7946851918","type":"ColumnDataSource"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"c3bfd26e-cc0f-448e-a0b8-ed8bac94dbc8","type":"Rect"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(287.287402, 292.075525]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[0.0],"label":["(287.287402, 292.075525]"],"line_alpha":[1.0],"line_color":["black"],"width":[4.788123364486012],"x":["289.6814635"],"y":[0.0]}},"id":"6198d6d6-56ad-4d53-9b24-58a56a70c5be","type":"ColumnDataSource"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(4.788123, 9.576247]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[313.0],"label":["(4.788123, 9.576247]"],"line_alpha":[1.0],"line_color":["black"],"width":[4.7881233644859815],"x":["7.1821850000000005"],"y":[156.5]}},"id":"d87704b5-c3ba-4f3a-a0fb-486507bbe9d1","type":"ColumnDataSource"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"2c8580c7-4cfd-4337-873f-80d865f4dc79","type":"Rect"},{"attributes":{"data_source":{"id":"c72d7375-5912-4bad-9c19-77befe97d3cc","type":"ColumnDataSource"},"glyph":{"id":"1141ac64-e908-452d-8ac6-16f4dbda59a6","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"9071ba82-aeff-4846-b1f2-5178fb231726","type":"GlyphRenderer"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"54755fd8-5d18-4353-b806-cd74b1b1e371","type":"Rect"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"0f1abf8f-a366-4eac-86d1-01e0288447ad","type":"Rect"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(282.499279, 287.287402]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[0.0],"label":["(282.499279, 287.287402]"],"line_alpha":[1.0],"line_color":["black"],"width":[4.788123364485955],"x":["284.8933405"],"y":[0.0]}},"id":"7c26cf5a-a7cc-4a90-a470-f0b42d15b5b0","type":"ColumnDataSource"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"962343e9-8d2c-4923-a28c-786082ebee0b","type":"Rect"},{"attributes":{"data_source":{"id":"859b1702-a075-40c7-ae7e-3eb5a5ea504f","type":"ColumnDataSource"},"glyph":{"id":"962343e9-8d2c-4923-a28c-786082ebee0b","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"c4d894fd-8859-4264-a1f5-bbec3258ac2b","type":"GlyphRenderer"},{"attributes":{"data_source":{"id":"2130cdc3-5a6b-409a-a25f-b7ca5365b17d","type":"ColumnDataSource"},"glyph":{"id":"2c8580c7-4cfd-4337-873f-80d865f4dc79","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"ab7d6c5c-5466-4c59-9062-ba8bada4d739","type":"GlyphRenderer"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"fcf8ea65-6fcc-4a2c-9f86-2a626136f16c","type":"Rect"},{"attributes":{"data_source":{"id":"d69a780b-973d-45aa-9fde-bf7946851918","type":"ColumnDataSource"},"glyph":{"id":"eb4ee46f-0ccd-4fcb-a23d-c3fa7cdc8f0c","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"eba94e39-cc54-4000-b69e-b29571b6a099","type":"GlyphRenderer"},{"attributes":{"data_source":{"id":"d87704b5-c3ba-4f3a-a0fb-486507bbe9d1","type":"ColumnDataSource"},"glyph":{"id":"54755fd8-5d18-4353-b806-cd74b1b1e371","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"38347de4-0cc2-443d-bcf7-5493e3e55656","type":"GlyphRenderer"},{"attributes":{"data_source":{"id":"911713b9-186d-4368-b67e-e104ec069612","type":"ColumnDataSource"},"glyph":{"id":"c3bfd26e-cc0f-448e-a0b8-ed8bac94dbc8","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"8e3c2fe4-78fc-4339-aeb9-ebe81010e4f1","type":"GlyphRenderer"}],"root_ids":["69634d55-8bb5-4411-9dcb-fa5caf7293df"]},"title":"Bokeh Application","version":"0.12.5"}};
            var render_items = [{"docid":"e5d6f7ce-715d-43c8-9c24-63407046fd51","elementid":"fc391734-8a4e-4dc3-a2ed-22dc942287e7","modelid":"69634d55-8bb5-4411-9dcb-fa5caf7293df"}];
            
            Bokeh.embed.embed_items(docs_json, render_items);
          };
          if (document.readyState != "loading") fn();
          else document.addEventListener("DOMContentLoaded", fn);
        })();
      },
      function(Bokeh) {
      }
    ];
  
    function run_inline_js() {
      
      if ((window.Bokeh !== undefined) || (force === true)) {
        for (var i = 0; i < inline_js.length; i++) {
          inline_js[i](window.Bokeh);
        }if (force === true) {
          display_loaded();
        }} else if (Date.now() < window._bokeh_timeout) {
        setTimeout(run_inline_js, 100);
      } else if (!window._bokeh_failed_load) {
        console.log("Bokeh: BokehJS failed to load within specified timeout.");
        window._bokeh_failed_load = true;
      } else if (force !== true) {
        var cell = $(document.getElementById("fc391734-8a4e-4dc3-a2ed-22dc942287e7")).parents('.cell').data().cell;
        cell.output_area.append_execute_result(NB_LOAD_WARNING)
      }
  
    }
  
    if (window._bokeh_is_loading === 0) {
      console.log("Bokeh: BokehJS loaded, going straight to plotting");
      run_inline_js();
    } else {
      load_libs(js_urls, function() {
        console.log("Bokeh: BokehJS plotting callback run at", now());
        run_inline_js();
      });
    }
  }(this));
</script>



```python
from bokeh.charts import Histogram
p = Histogram(train_df["Survived"], palette=['red','blue'])
show(p)
```




    <div class="bk-root">
        <div class="bk-plotdiv" id="02de312b-2ebb-4a03-a01f-c6b034f0cbe2"></div>
    </div>
<script type="text/javascript">
  
  (function(global) {
    function now() {
      return new Date();
    }
  
    var force = false;
  
    if (typeof (window._bokeh_onload_callbacks) === "undefined" || force === true) {
      window._bokeh_onload_callbacks = [];
      window._bokeh_is_loading = undefined;
    }
  
  
    
    if (typeof (window._bokeh_timeout) === "undefined" || force === true) {
      window._bokeh_timeout = Date.now() + 0;
      window._bokeh_failed_load = false;
    }
  
    var NB_LOAD_WARNING = {'data': {'text/html':
       "<div style='background-color: #fdd'>\n"+
       "<p>\n"+
       "BokehJS does not appear to have successfully loaded. If loading BokehJS from CDN, this \n"+
       "may be due to a slow or bad network connection. Possible fixes:\n"+
       "</p>\n"+
       "<ul>\n"+
       "<li>re-rerun `output_notebook()` to attempt to load from CDN again, or</li>\n"+
       "<li>use INLINE resources instead, as so:</li>\n"+
       "</ul>\n"+
       "<code>\n"+
       "from bokeh.resources import INLINE\n"+
       "output_notebook(resources=INLINE)\n"+
       "</code>\n"+
       "</div>"}};
  
    function display_loaded() {
      if (window.Bokeh !== undefined) {
        var el = document.getElementById("02de312b-2ebb-4a03-a01f-c6b034f0cbe2");
        el.textContent = "BokehJS " + Bokeh.version + " successfully loaded.";
      } else if (Date.now() < window._bokeh_timeout) {
        setTimeout(display_loaded, 100)
      }
    }
  
    function run_callbacks() {
      window._bokeh_onload_callbacks.forEach(function(callback) { callback() });
      delete window._bokeh_onload_callbacks
      console.info("Bokeh: all callbacks have finished");
    }
  
    function load_libs(js_urls, callback) {
      window._bokeh_onload_callbacks.push(callback);
      if (window._bokeh_is_loading > 0) {
        console.log("Bokeh: BokehJS is being loaded, scheduling callback at", now());
        return null;
      }
      if (js_urls == null || js_urls.length === 0) {
        run_callbacks();
        return null;
      }
      console.log("Bokeh: BokehJS not loaded, scheduling load and callback at", now());
      window._bokeh_is_loading = js_urls.length;
      for (var i = 0; i < js_urls.length; i++) {
        var url = js_urls[i];
        var s = document.createElement('script');
        s.src = url;
        s.async = false;
        s.onreadystatechange = s.onload = function() {
          window._bokeh_is_loading--;
          if (window._bokeh_is_loading === 0) {
            console.log("Bokeh: all BokehJS libraries loaded");
            run_callbacks()
          }
        };
        s.onerror = function() {
          console.warn("failed to load library " + url);
        };
        console.log("Bokeh: injecting script tag for BokehJS library: ", url);
        document.getElementsByTagName("head")[0].appendChild(s);
      }
    };var element = document.getElementById("02de312b-2ebb-4a03-a01f-c6b034f0cbe2");
    if (element == null) {
      console.log("Bokeh: ERROR: autoload.js configured with elementid '02de312b-2ebb-4a03-a01f-c6b034f0cbe2' but no matching script tag was found. ")
      return false;
    }
  
    var js_urls = [];
  
    var inline_js = [
      function(Bokeh) {
        (function() {
          var fn = function() {
            var docs_json = {"6f797d21-f557-4dcc-ab7a-947a185cb906":{"roots":{"references":[{"attributes":{"data_source":{"id":"289a41fc-436c-437f-ade0-7bc6c3264a0b","type":"ColumnDataSource"},"glyph":{"id":"32330b92-19b8-48b5-9c3c-23e61311e269","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"82a97d45-9c44-419f-8753-4e90be67434c","type":"GlyphRenderer"},{"attributes":{"location":"top_left","plot":{"id":"34333a5b-e40f-4deb-b2fc-44cdc2242c80","subtype":"Chart","type":"Plot"}},"id":"4a1eb0d9-1488-4fc3-ae63-784ee3c026d2","type":"Legend"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(0.400000, 0.600000]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[0.0],"label":["(0.400000, 0.600000]"],"line_alpha":[1.0],"line_color":["black"],"width":[0.20000000000000007],"x":["0.5"],"y":[0.0]}},"id":"4d863e67-a5d0-47f7-9eb8-fe9fc46b483d","type":"ColumnDataSource"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"a436c7a3-47a2-4af2-882e-dbe2e8d466cb","type":"Rect"},{"attributes":{"data_source":{"id":"4d863e67-a5d0-47f7-9eb8-fe9fc46b483d","type":"ColumnDataSource"},"glyph":{"id":"a436c7a3-47a2-4af2-882e-dbe2e8d466cb","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"4428242f-77d5-41b1-a964-04f56e5c4a8e","type":"GlyphRenderer"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(0.600000, 0.800000]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[0.0],"label":["(0.600000, 0.800000]"],"line_alpha":[1.0],"line_color":["black"],"width":[0.19999999999999996],"x":["0.7"],"y":[0.0]}},"id":"79acf964-d594-4485-8881-6f1e7d6eec79","type":"ColumnDataSource"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"df5de8af-a60a-4229-9ccf-ba03e0b335a0","type":"Rect"},{"attributes":{"data_source":{"id":"deafbdf0-1893-49f2-aab4-b70ec96c1c42","type":"ColumnDataSource"},"glyph":{"id":"e81474f7-3ce4-4140-b194-4eded501583e","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"ee08869f-9eb1-495f-8c95-13eeedf1f732","type":"GlyphRenderer"},{"attributes":{"data_source":{"id":"79acf964-d594-4485-8881-6f1e7d6eec79","type":"ColumnDataSource"},"glyph":{"id":"df5de8af-a60a-4229-9ccf-ba03e0b335a0","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"ff12887f-0007-4183-a7cf-c1b4213810ad","type":"GlyphRenderer"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(0.800000, 1.000000]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[342.0],"label":["(0.800000, 1.000000]"],"line_alpha":[1.0],"line_color":["black"],"width":[0.19999999999999996],"x":["0.9"],"y":[171.0]}},"id":"7636b92e-4811-4356-b070-99ba1ca205a3","type":"ColumnDataSource"},{"attributes":{"plot":{"id":"34333a5b-e40f-4deb-b2fc-44cdc2242c80","subtype":"Chart","type":"Plot"}},"id":"08bdc27a-bd72-4329-b9c9-34cb5f721c42","type":"PanTool"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"ad61beb6-9d3f-41df-96ff-f1d88fe21fa5","type":"Rect"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["[0.000000, 0.200000]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[549.0],"label":["[0.000000, 0.200000]"],"line_alpha":[1.0],"line_color":["black"],"width":[0.2],"x":["0.1"],"y":[274.5]}},"id":"289a41fc-436c-437f-ade0-7bc6c3264a0b","type":"ColumnDataSource"},{"attributes":{},"id":"bed9c76c-b24b-484d-b2c7-c8df3331aa2e","type":"ToolEvents"},{"attributes":{"bottom_units":"screen","fill_alpha":{"value":0.5},"fill_color":{"value":"lightgrey"},"left_units":"screen","level":"overlay","line_alpha":{"value":1.0},"line_color":{"value":"black"},"line_dash":[4,4],"line_width":{"value":2},"plot":null,"render_mode":"css","right_units":"screen","top_units":"screen"},"id":"43d353e7-b71c-4fff-89aa-59b94ff3d985","type":"BoxAnnotation"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"e81474f7-3ce4-4140-b194-4eded501583e","type":"Rect"},{"attributes":{"active_drag":"auto","active_scroll":"auto","active_tap":"auto","tools":[{"id":"08bdc27a-bd72-4329-b9c9-34cb5f721c42","type":"PanTool"},{"id":"e09bb783-17fd-4b3d-9f37-c22f36552b94","type":"WheelZoomTool"},{"id":"28964be6-5f88-4ed5-ae14-fc58f088235b","type":"BoxZoomTool"},{"id":"b5286d71-5e25-49a8-9fcc-90086eaee5f6","type":"SaveTool"},{"id":"92d63eb0-7894-43ca-989a-05207033ed4e","type":"ResetTool"},{"id":"9194ae93-f9e0-483b-be18-01ecad10327b","type":"HelpTool"}]},"id":"9bf2b539-b67e-48ed-b444-816ba3712fb0","type":"Toolbar"},{"attributes":{"axis_label":"Survived","formatter":{"id":"90262f62-d6fc-400f-b154-507099c8b38e","type":"BasicTickFormatter"},"plot":{"id":"34333a5b-e40f-4deb-b2fc-44cdc2242c80","subtype":"Chart","type":"Plot"},"ticker":{"id":"f69d1577-9132-403a-b6a1-e37302344f28","type":"BasicTicker"}},"id":"cebdd4c7-130e-40c9-8569-df451ccd39a3","type":"LinearAxis"},{"attributes":{"callback":null,"end":1.025,"start":-0.025},"id":"98e7a667-d880-4048-9939-8e853ca47a46","type":"Range1d"},{"attributes":{"plot":null,"text":null},"id":"3615c347-d3b7-45bd-a07e-bce31bba1712","type":"Title"},{"attributes":{"data_source":{"id":"7636b92e-4811-4356-b070-99ba1ca205a3","type":"ColumnDataSource"},"glyph":{"id":"ad61beb6-9d3f-41df-96ff-f1d88fe21fa5","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"d44e851e-668c-456b-811a-762615dec22e","type":"GlyphRenderer"},{"attributes":{"axis_label":"Count( Survived )","formatter":{"id":"c6046229-23a2-49fd-a870-f6be2653dfa8","type":"BasicTickFormatter"},"plot":{"id":"34333a5b-e40f-4deb-b2fc-44cdc2242c80","subtype":"Chart","type":"Plot"},"ticker":{"id":"deb76567-a200-4875-94d3-e5e077d79d2a","type":"BasicTicker"}},"id":"55981d9d-2b66-42e8-ad42-1f5f451a55d7","type":"LinearAxis"},{"attributes":{"below":[{"id":"cebdd4c7-130e-40c9-8569-df451ccd39a3","type":"LinearAxis"}],"css_classes":null,"left":[{"id":"55981d9d-2b66-42e8-ad42-1f5f451a55d7","type":"LinearAxis"}],"renderers":[{"id":"43d353e7-b71c-4fff-89aa-59b94ff3d985","type":"BoxAnnotation"},{"id":"82a97d45-9c44-419f-8753-4e90be67434c","type":"GlyphRenderer"},{"id":"ee08869f-9eb1-495f-8c95-13eeedf1f732","type":"GlyphRenderer"},{"id":"4428242f-77d5-41b1-a964-04f56e5c4a8e","type":"GlyphRenderer"},{"id":"ff12887f-0007-4183-a7cf-c1b4213810ad","type":"GlyphRenderer"},{"id":"d44e851e-668c-456b-811a-762615dec22e","type":"GlyphRenderer"},{"id":"4a1eb0d9-1488-4fc3-ae63-784ee3c026d2","type":"Legend"},{"id":"cebdd4c7-130e-40c9-8569-df451ccd39a3","type":"LinearAxis"},{"id":"55981d9d-2b66-42e8-ad42-1f5f451a55d7","type":"LinearAxis"},{"id":"6c353285-8836-4fb7-b786-46b47a904908","type":"Grid"}],"title":{"id":"3615c347-d3b7-45bd-a07e-bce31bba1712","type":"Title"},"tool_events":{"id":"bed9c76c-b24b-484d-b2c7-c8df3331aa2e","type":"ToolEvents"},"toolbar":{"id":"9bf2b539-b67e-48ed-b444-816ba3712fb0","type":"Toolbar"},"x_mapper_type":"auto","x_range":{"id":"98e7a667-d880-4048-9939-8e853ca47a46","type":"Range1d"},"y_mapper_type":"auto","y_range":{"id":"304a0e02-d1b5-4790-9885-5707577e732b","type":"Range1d"}},"id":"34333a5b-e40f-4deb-b2fc-44cdc2242c80","subtype":"Chart","type":"Plot"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(0.200000, 0.400000]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[0.0],"label":["(0.200000, 0.400000]"],"line_alpha":[1.0],"line_color":["black"],"width":[0.2],"x":["0.30000000000000004"],"y":[0.0]}},"id":"deafbdf0-1893-49f2-aab4-b70ec96c1c42","type":"ColumnDataSource"},{"attributes":{},"id":"deb76567-a200-4875-94d3-e5e077d79d2a","type":"BasicTicker"},{"attributes":{"dimension":1,"plot":{"id":"34333a5b-e40f-4deb-b2fc-44cdc2242c80","subtype":"Chart","type":"Plot"},"ticker":{"id":"deb76567-a200-4875-94d3-e5e077d79d2a","type":"BasicTicker"}},"id":"6c353285-8836-4fb7-b786-46b47a904908","type":"Grid"},{"attributes":{"plot":{"id":"34333a5b-e40f-4deb-b2fc-44cdc2242c80","subtype":"Chart","type":"Plot"}},"id":"e09bb783-17fd-4b3d-9f37-c22f36552b94","type":"WheelZoomTool"},{"attributes":{"callback":null,"end":603.9000000000001},"id":"304a0e02-d1b5-4790-9885-5707577e732b","type":"Range1d"},{"attributes":{"overlay":{"id":"43d353e7-b71c-4fff-89aa-59b94ff3d985","type":"BoxAnnotation"},"plot":{"id":"34333a5b-e40f-4deb-b2fc-44cdc2242c80","subtype":"Chart","type":"Plot"}},"id":"28964be6-5f88-4ed5-ae14-fc58f088235b","type":"BoxZoomTool"},{"attributes":{"plot":{"id":"34333a5b-e40f-4deb-b2fc-44cdc2242c80","subtype":"Chart","type":"Plot"}},"id":"92d63eb0-7894-43ca-989a-05207033ed4e","type":"ResetTool"},{"attributes":{"plot":{"id":"34333a5b-e40f-4deb-b2fc-44cdc2242c80","subtype":"Chart","type":"Plot"}},"id":"b5286d71-5e25-49a8-9fcc-90086eaee5f6","type":"SaveTool"},{"attributes":{"plot":{"id":"34333a5b-e40f-4deb-b2fc-44cdc2242c80","subtype":"Chart","type":"Plot"}},"id":"9194ae93-f9e0-483b-be18-01ecad10327b","type":"HelpTool"},{"attributes":{},"id":"c6046229-23a2-49fd-a870-f6be2653dfa8","type":"BasicTickFormatter"},{"attributes":{},"id":"90262f62-d6fc-400f-b154-507099c8b38e","type":"BasicTickFormatter"},{"attributes":{},"id":"f69d1577-9132-403a-b6a1-e37302344f28","type":"BasicTicker"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"32330b92-19b8-48b5-9c3c-23e61311e269","type":"Rect"}],"root_ids":["34333a5b-e40f-4deb-b2fc-44cdc2242c80"]},"title":"Bokeh Application","version":"0.12.5"}};
            var render_items = [{"docid":"6f797d21-f557-4dcc-ab7a-947a185cb906","elementid":"02de312b-2ebb-4a03-a01f-c6b034f0cbe2","modelid":"34333a5b-e40f-4deb-b2fc-44cdc2242c80"}];
            
            Bokeh.embed.embed_items(docs_json, render_items);
          };
          if (document.readyState != "loading") fn();
          else document.addEventListener("DOMContentLoaded", fn);
        })();
      },
      function(Bokeh) {
      }
    ];
  
    function run_inline_js() {
      
      if ((window.Bokeh !== undefined) || (force === true)) {
        for (var i = 0; i < inline_js.length; i++) {
          inline_js[i](window.Bokeh);
        }if (force === true) {
          display_loaded();
        }} else if (Date.now() < window._bokeh_timeout) {
        setTimeout(run_inline_js, 100);
      } else if (!window._bokeh_failed_load) {
        console.log("Bokeh: BokehJS failed to load within specified timeout.");
        window._bokeh_failed_load = true;
      } else if (force !== true) {
        var cell = $(document.getElementById("02de312b-2ebb-4a03-a01f-c6b034f0cbe2")).parents('.cell').data().cell;
        cell.output_area.append_execute_result(NB_LOAD_WARNING)
      }
  
    }
  
    if (window._bokeh_is_loading === 0) {
      console.log("Bokeh: BokehJS loaded, going straight to plotting");
      run_inline_js();
    } else {
      load_libs(js_urls, function() {
        console.log("Bokeh: BokehJS plotting callback run at", now());
        run_inline_js();
      });
    }
  }(this));
</script>



```python
from bokeh.charts import Histogram
p = Histogram(train_df["SibSp"], palette=['red','blue'])
show(p)
```




    <div class="bk-root">
        <div class="bk-plotdiv" id="ae0cf855-51c2-406e-a829-bd14a146ba08"></div>
    </div>
<script type="text/javascript">
  
  (function(global) {
    function now() {
      return new Date();
    }
  
    var force = false;
  
    if (typeof (window._bokeh_onload_callbacks) === "undefined" || force === true) {
      window._bokeh_onload_callbacks = [];
      window._bokeh_is_loading = undefined;
    }
  
  
    
    if (typeof (window._bokeh_timeout) === "undefined" || force === true) {
      window._bokeh_timeout = Date.now() + 0;
      window._bokeh_failed_load = false;
    }
  
    var NB_LOAD_WARNING = {'data': {'text/html':
       "<div style='background-color: #fdd'>\n"+
       "<p>\n"+
       "BokehJS does not appear to have successfully loaded. If loading BokehJS from CDN, this \n"+
       "may be due to a slow or bad network connection. Possible fixes:\n"+
       "</p>\n"+
       "<ul>\n"+
       "<li>re-rerun `output_notebook()` to attempt to load from CDN again, or</li>\n"+
       "<li>use INLINE resources instead, as so:</li>\n"+
       "</ul>\n"+
       "<code>\n"+
       "from bokeh.resources import INLINE\n"+
       "output_notebook(resources=INLINE)\n"+
       "</code>\n"+
       "</div>"}};
  
    function display_loaded() {
      if (window.Bokeh !== undefined) {
        var el = document.getElementById("ae0cf855-51c2-406e-a829-bd14a146ba08");
        el.textContent = "BokehJS " + Bokeh.version + " successfully loaded.";
      } else if (Date.now() < window._bokeh_timeout) {
        setTimeout(display_loaded, 100)
      }
    }
  
    function run_callbacks() {
      window._bokeh_onload_callbacks.forEach(function(callback) { callback() });
      delete window._bokeh_onload_callbacks
      console.info("Bokeh: all callbacks have finished");
    }
  
    function load_libs(js_urls, callback) {
      window._bokeh_onload_callbacks.push(callback);
      if (window._bokeh_is_loading > 0) {
        console.log("Bokeh: BokehJS is being loaded, scheduling callback at", now());
        return null;
      }
      if (js_urls == null || js_urls.length === 0) {
        run_callbacks();
        return null;
      }
      console.log("Bokeh: BokehJS not loaded, scheduling load and callback at", now());
      window._bokeh_is_loading = js_urls.length;
      for (var i = 0; i < js_urls.length; i++) {
        var url = js_urls[i];
        var s = document.createElement('script');
        s.src = url;
        s.async = false;
        s.onreadystatechange = s.onload = function() {
          window._bokeh_is_loading--;
          if (window._bokeh_is_loading === 0) {
            console.log("Bokeh: all BokehJS libraries loaded");
            run_callbacks()
          }
        };
        s.onerror = function() {
          console.warn("failed to load library " + url);
        };
        console.log("Bokeh: injecting script tag for BokehJS library: ", url);
        document.getElementsByTagName("head")[0].appendChild(s);
      }
    };var element = document.getElementById("ae0cf855-51c2-406e-a829-bd14a146ba08");
    if (element == null) {
      console.log("Bokeh: ERROR: autoload.js configured with elementid 'ae0cf855-51c2-406e-a829-bd14a146ba08' but no matching script tag was found. ")
      return false;
    }
  
    var js_urls = [];
  
    var inline_js = [
      function(Bokeh) {
        (function() {
          var fn = function() {
            var docs_json = {"1c624a9d-6b1b-44a0-9e68-7e10bb22623c":{"roots":{"references":[{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"b6eecd59-0043-4add-9147-cfcb84a59b65","type":"Rect"},{"attributes":{"data_source":{"id":"63825bd2-ab45-4a6c-8afa-61499240c757","type":"ColumnDataSource"},"glyph":{"id":"ab6dc795-40d5-4331-a89f-e61db59f575b","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"eecd62b0-f669-40b0-adde-95a847739c0e","type":"GlyphRenderer"},{"attributes":{"location":"top_left","plot":{"id":"0a4ac7b2-5e5d-4da6-b840-f6945579b6bf","subtype":"Chart","type":"Plot"}},"id":"a9874103-120a-4060-ab37-7d0f9a86c826","type":"Legend"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"3a3bc753-f7b5-4cc7-be1b-6008ac4eeef7","type":"Rect"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(0.615385, 0.820513]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[0.0],"label":["(0.615385, 0.820513]"],"line_alpha":[1.0],"line_color":["black"],"width":[0.20512820512820507],"x":["0.717949"],"y":[0.0]}},"id":"1e7b52d5-53db-47db-90a6-d9fd87df5e5b","type":"ColumnDataSource"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(1.435897, 1.641026]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[0.0],"label":["(1.435897, 1.641026]"],"line_alpha":[1.0],"line_color":["black"],"width":[0.20512820512820507],"x":["1.5384615"],"y":[0.0]}},"id":"a484ae0b-70e6-40d4-a202-26081c74a3df","type":"ColumnDataSource"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"29487e25-fc7d-46da-9ef6-4d53082ae87a","type":"Rect"},{"attributes":{"data_source":{"id":"7ea2baea-3a85-411d-84e3-4e2149cead1b","type":"ColumnDataSource"},"glyph":{"id":"7d9bcbf6-cc51-41e8-8417-8b8fd7439e7e","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"ddbcee31-2763-475e-a7fb-642a1f6fd16d","type":"GlyphRenderer"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(0.410256, 0.615385]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[0.0],"label":["(0.410256, 0.615385]"],"line_alpha":[1.0],"line_color":["black"],"width":[0.20512820512820518],"x":["0.5128205"],"y":[0.0]}},"id":"406be2dc-2ee3-4243-8da8-b7e770bc3064","type":"ColumnDataSource"},{"attributes":{"data_source":{"id":"406be2dc-2ee3-4243-8da8-b7e770bc3064","type":"ColumnDataSource"},"glyph":{"id":"e8b9cabd-910d-4520-8e6a-aed2675c2aeb","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"290352f2-e878-4f72-93e6-b8a325155280","type":"GlyphRenderer"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"c191d18f-43d1-4442-a16c-e12d6ee00a9b","type":"Rect"},{"attributes":{"data_source":{"id":"84d62263-dc11-43c8-98e9-8c8759f9aef3","type":"ColumnDataSource"},"glyph":{"id":"82ea0f5e-ea45-47bb-bd15-5e30a6bb11da","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"784cba28-f4dd-4490-a6e7-9167e95004a4","type":"GlyphRenderer"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(0.820513, 1.025641]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[209.0],"label":["(0.820513, 1.025641]"],"line_alpha":[1.0],"line_color":["black"],"width":[0.20512820512820507],"x":["0.923077"],"y":[104.5]}},"id":"e6554330-299b-4869-ad5f-9164ee340a89","type":"ColumnDataSource"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(2.461538, 2.666667]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[0.0],"label":["(2.461538, 2.666667]"],"line_alpha":[1.0],"line_color":["black"],"width":[0.20512820512820484],"x":["2.5641024999999997"],"y":[0.0]}},"id":"a9af4b36-717a-4c18-ac67-f3567f3ecdbf","type":"ColumnDataSource"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"2474ff56-8b9b-4ea9-8b01-7c34ca74d15d","type":"Rect"},{"attributes":{"data_source":{"id":"1e7b52d5-53db-47db-90a6-d9fd87df5e5b","type":"ColumnDataSource"},"glyph":{"id":"b6eecd59-0043-4add-9147-cfcb84a59b65","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"a488a41f-850e-40e4-ba02-57765921b01d","type":"GlyphRenderer"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(5.538462, 5.743590]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[0.0],"label":["(5.538462, 5.743590]"],"line_alpha":[1.0],"line_color":["black"],"width":[0.2051282051282053],"x":["5.641026"],"y":[0.0]}},"id":"d2b4931e-bc14-4fe0-970b-9573e7f82d3b","type":"ColumnDataSource"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"477409a9-6e79-41f2-9a97-5c8dcf062ad6","type":"Rect"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(4.923077, 5.128205]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[5.0],"label":["(4.923077, 5.128205]"],"line_alpha":[1.0],"line_color":["black"],"width":[0.2051282051282044],"x":["5.025641"],"y":[2.5]}},"id":"7ea2baea-3a85-411d-84e3-4e2149cead1b","type":"ColumnDataSource"},{"attributes":{"data_source":{"id":"d2b4931e-bc14-4fe0-970b-9573e7f82d3b","type":"ColumnDataSource"},"glyph":{"id":"1bf79e1e-bb02-487e-b19b-e5cb2c45f9d4","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"1dccdccc-240a-41fb-94ba-3f03821aa415","type":"GlyphRenderer"},{"attributes":{"data_source":{"id":"a9af4b36-717a-4c18-ac67-f3567f3ecdbf","type":"ColumnDataSource"},"glyph":{"id":"2353d042-a58d-467f-9967-197d9e3107a5","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"cd03ba3a-14a5-41e1-ac01-bfb6d1c5c0e1","type":"GlyphRenderer"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(1.025641, 1.230769]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[0.0],"label":["(1.025641, 1.230769]"],"line_alpha":[1.0],"line_color":["black"],"width":[0.2051282051282053],"x":["1.128205"],"y":[0.0]}},"id":"fad45a87-8da7-41fa-adc9-41f9fed340ce","type":"ColumnDataSource"},{"attributes":{"data_source":{"id":"e6554330-299b-4869-ad5f-9164ee340a89","type":"ColumnDataSource"},"glyph":{"id":"c191d18f-43d1-4442-a16c-e12d6ee00a9b","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"b9525b2d-62c4-4483-9ff2-4640f585dc04","type":"GlyphRenderer"},{"attributes":{"data_source":{"id":"466cff73-b0f2-43a9-b9a7-a8dfe6397719","type":"ColumnDataSource"},"glyph":{"id":"29487e25-fc7d-46da-9ef6-4d53082ae87a","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"99b16abe-6ca2-4b16-af77-8c8af6df31db","type":"GlyphRenderer"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"75d3b9f1-0de3-43d4-b747-00ab9f6f6774","type":"Rect"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(4.717949, 4.923077]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[0.0],"label":["(4.717949, 4.923077]"],"line_alpha":[1.0],"line_color":["black"],"width":[0.2051282051282053],"x":["4.820513"],"y":[0.0]}},"id":"63825bd2-ab45-4a6c-8afa-61499240c757","type":"ColumnDataSource"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(1.230769, 1.435897]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[0.0],"label":["(1.230769, 1.435897]"],"line_alpha":[1.0],"line_color":["black"],"width":[0.20512820512820507],"x":["1.333333"],"y":[0.0]}},"id":"e4d58a42-08c4-4a87-9de3-561258486c95","type":"ColumnDataSource"},{"attributes":{"data_source":{"id":"790ea2b4-0c4c-4551-8e13-32d32b7d541d","type":"ColumnDataSource"},"glyph":{"id":"a11da88e-c6ab-429a-8842-df93defdf5f1","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"1906d171-4903-4edb-b977-a3b1afcb5856","type":"GlyphRenderer"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(6.564103, 6.769231]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[0.0],"label":["(6.564103, 6.769231]"],"line_alpha":[1.0],"line_color":["black"],"width":[0.2051282051282053],"x":["6.666667"],"y":[0.0]}},"id":"412fb63e-120f-4f82-ad28-f69acf339ca3","type":"ColumnDataSource"},{"attributes":{"data_source":{"id":"fad45a87-8da7-41fa-adc9-41f9fed340ce","type":"ColumnDataSource"},"glyph":{"id":"477409a9-6e79-41f2-9a97-5c8dcf062ad6","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"e3d26113-1ab0-4cd3-b6f9-8370a39d66e0","type":"GlyphRenderer"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"6d93c4a7-1235-4e23-a248-4941a225b896","type":"Rect"},{"attributes":{"data_source":{"id":"2355a989-ed4c-4a3d-8211-5b4098898954","type":"ColumnDataSource"},"glyph":{"id":"30cde64d-6e3b-4ee9-97bd-dfe27f9165ff","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"7217bb36-9452-40d5-961b-fcffeee652c0","type":"GlyphRenderer"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"2353d042-a58d-467f-9967-197d9e3107a5","type":"Rect"},{"attributes":{"data_source":{"id":"1cba1185-1a77-4499-aeb3-850b1382e05d","type":"ColumnDataSource"},"glyph":{"id":"316e501f-173c-4275-a846-2b95a6b5b476","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"b048964f-b171-4fd8-bc4a-bf11f17b9c03","type":"GlyphRenderer"},{"attributes":{},"id":"e1ac8b98-1c00-402d-b601-83a29ec3e590","type":"ToolEvents"},{"attributes":{"below":[{"id":"f38a4b37-ceaa-4d70-a661-5ade3222c176","type":"LinearAxis"}],"css_classes":null,"left":[{"id":"49c493b5-2878-4ba7-86f6-12e0b7ee6563","type":"LinearAxis"}],"renderers":[{"id":"ef082280-6b7e-4146-98bb-29a1fb3c09e8","type":"BoxAnnotation"},{"id":"51451761-ab2a-41c8-a5d0-e78fa4a85f1a","type":"GlyphRenderer"},{"id":"9fafb47a-6846-4f81-82c4-c632c735f28b","type":"GlyphRenderer"},{"id":"290352f2-e878-4f72-93e6-b8a325155280","type":"GlyphRenderer"},{"id":"a488a41f-850e-40e4-ba02-57765921b01d","type":"GlyphRenderer"},{"id":"b9525b2d-62c4-4483-9ff2-4640f585dc04","type":"GlyphRenderer"},{"id":"e3d26113-1ab0-4cd3-b6f9-8370a39d66e0","type":"GlyphRenderer"},{"id":"261de225-eecf-4e09-bcd9-33e28bd85fc1","type":"GlyphRenderer"},{"id":"ff9711ab-e06c-4d79-a9a5-4c0b42454805","type":"GlyphRenderer"},{"id":"3ef5e01c-a294-476f-bc56-7e85c7f399b1","type":"GlyphRenderer"},{"id":"32b1eafe-9256-48d0-b7f1-8470a2de533c","type":"GlyphRenderer"},{"id":"b048964f-b171-4fd8-bc4a-bf11f17b9c03","type":"GlyphRenderer"},{"id":"784cba28-f4dd-4490-a6e7-9167e95004a4","type":"GlyphRenderer"},{"id":"cd03ba3a-14a5-41e1-ac01-bfb6d1c5c0e1","type":"GlyphRenderer"},{"id":"3cfeccb9-d68e-4c9e-a6f2-215fadc9b62b","type":"GlyphRenderer"},{"id":"97e5e6f0-aa11-4fb5-9b71-724239010e51","type":"GlyphRenderer"},{"id":"99b16abe-6ca2-4b16-af77-8c8af6df31db","type":"GlyphRenderer"},{"id":"e1a96def-7fa1-46a2-80de-65c076aac7c4","type":"GlyphRenderer"},{"id":"e7ce14fa-6316-413f-b3d9-75e8a73922d0","type":"GlyphRenderer"},{"id":"e46deb94-9cff-484f-94de-f19c47d9e332","type":"GlyphRenderer"},{"id":"71f19382-28f7-497b-8fa4-d5d857287cc4","type":"GlyphRenderer"},{"id":"7217bb36-9452-40d5-961b-fcffeee652c0","type":"GlyphRenderer"},{"id":"1906d171-4903-4edb-b977-a3b1afcb5856","type":"GlyphRenderer"},{"id":"663af8a1-75d0-49c0-abb8-8541e3ef33bd","type":"GlyphRenderer"},{"id":"eecd62b0-f669-40b0-adde-95a847739c0e","type":"GlyphRenderer"},{"id":"ddbcee31-2763-475e-a7fb-642a1f6fd16d","type":"GlyphRenderer"},{"id":"168d1e47-5666-4f68-9f5c-63f50a12c9bf","type":"GlyphRenderer"},{"id":"0c0e3cda-9e62-4aa8-9939-9681d5a410a9","type":"GlyphRenderer"},{"id":"1dccdccc-240a-41fb-94ba-3f03821aa415","type":"GlyphRenderer"},{"id":"599b20ee-3a2c-44bb-86fc-2b2ba3d4639a","type":"GlyphRenderer"},{"id":"bc6ad9a2-b4f3-46e9-b0b6-c30e067bc48e","type":"GlyphRenderer"},{"id":"e3367e2b-eca5-42ed-810f-195f18343bbb","type":"GlyphRenderer"},{"id":"a64705ea-b2c8-4de6-93da-812e6444aa29","type":"GlyphRenderer"},{"id":"9558bf13-afe8-4d65-bb33-e797796f7d01","type":"GlyphRenderer"},{"id":"db56dc85-7f38-408f-a297-153f44e7e5f2","type":"GlyphRenderer"},{"id":"eb80e16b-6fae-46a5-a511-2364f6848898","type":"GlyphRenderer"},{"id":"7cae2a19-83a2-44a9-926e-e4bb48898589","type":"GlyphRenderer"},{"id":"b97165b1-f57c-4f6c-bca8-58a61ee7adac","type":"GlyphRenderer"},{"id":"517caff9-b39b-4154-8970-8fa6b45fa17e","type":"GlyphRenderer"},{"id":"a4e92e21-c4d5-494d-86d6-fd61d2828065","type":"GlyphRenderer"},{"id":"a9874103-120a-4060-ab37-7d0f9a86c826","type":"Legend"},{"id":"f38a4b37-ceaa-4d70-a661-5ade3222c176","type":"LinearAxis"},{"id":"49c493b5-2878-4ba7-86f6-12e0b7ee6563","type":"LinearAxis"},{"id":"965ec388-bb8f-466f-80b0-14546b28b2dd","type":"Grid"}],"title":{"id":"5c906e34-13bc-4c43-a18f-fd634b2b873c","type":"Title"},"tool_events":{"id":"e1ac8b98-1c00-402d-b601-83a29ec3e590","type":"ToolEvents"},"toolbar":{"id":"d69481dc-0caa-4a00-a3c2-952b615cfebd","type":"Toolbar"},"x_mapper_type":"auto","x_range":{"id":"1c39fc75-413d-4db5-9e85-2a579978cd83","type":"Range1d"},"y_mapper_type":"auto","y_range":{"id":"dcff1790-ffca-40b5-8052-4b694ac03e40","type":"Range1d"}},"id":"0a4ac7b2-5e5d-4da6-b840-f6945579b6bf","subtype":"Chart","type":"Plot"},{"attributes":{"data_source":{"id":"e4d58a42-08c4-4a87-9de3-561258486c95","type":"ColumnDataSource"},"glyph":{"id":"75d3b9f1-0de3-43d4-b747-00ab9f6f6774","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"261de225-eecf-4e09-bcd9-33e28bd85fc1","type":"GlyphRenderer"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"82ea0f5e-ea45-47bb-bd15-5e30a6bb11da","type":"Rect"},{"attributes":{"callback":null,"end":8.200000102564102,"start":-0.20000010256410256},"id":"1c39fc75-413d-4db5-9e85-2a579978cd83","type":"Range1d"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(5.743590, 5.948718]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[0.0],"label":["(5.743590, 5.948718]"],"line_alpha":[1.0],"line_color":["black"],"width":[0.2051282051282053],"x":["5.846154"],"y":[0.0]}},"id":"2a6cdb74-4867-403c-8676-8ef0134ca660","type":"ColumnDataSource"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"7d9bcbf6-cc51-41e8-8417-8b8fd7439e7e","type":"Rect"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"68105310-8c15-4b9d-80a2-51536a2c823c","type":"Rect"},{"attributes":{"plot":null,"text":null},"id":"5c906e34-13bc-4c43-a18f-fd634b2b873c","type":"Title"},{"attributes":{"active_drag":"auto","active_scroll":"auto","active_tap":"auto","tools":[{"id":"cd02139d-763e-42ae-a6c1-2f463e7a82fc","type":"PanTool"},{"id":"b6f173a1-9d95-4e6f-93c1-206c160fd5f0","type":"WheelZoomTool"},{"id":"6cd6c5f6-8dd1-43de-9764-58ceb6e3c7ff","type":"BoxZoomTool"},{"id":"7446428d-50be-4983-bd78-fa3d9864f12e","type":"SaveTool"},{"id":"5ea443c5-a1e5-4c06-9ad7-434097c35175","type":"ResetTool"},{"id":"3f0e3052-1e18-4762-bdd0-b34cba7bed1b","type":"HelpTool"}]},"id":"d69481dc-0caa-4a00-a3c2-952b615cfebd","type":"Toolbar"},{"attributes":{"bottom_units":"screen","fill_alpha":{"value":0.5},"fill_color":{"value":"lightgrey"},"left_units":"screen","level":"overlay","line_alpha":{"value":1.0},"line_color":{"value":"black"},"line_dash":[4,4],"line_width":{"value":2},"plot":null,"render_mode":"css","right_units":"screen","top_units":"screen"},"id":"ef082280-6b7e-4146-98bb-29a1fb3c09e8","type":"BoxAnnotation"},{"attributes":{"callback":null,"end":668.8000000000001},"id":"dcff1790-ffca-40b5-8052-4b694ac03e40","type":"Range1d"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(4.307692, 4.512821]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[0.0],"label":["(4.307692, 4.512821]"],"line_alpha":[1.0],"line_color":["black"],"width":[0.2051282051282053],"x":["4.4102565"],"y":[0.0]}},"id":"790ea2b4-0c4c-4551-8e13-32d32b7d541d","type":"ColumnDataSource"},{"attributes":{"plot":{"id":"0a4ac7b2-5e5d-4da6-b840-f6945579b6bf","subtype":"Chart","type":"Plot"}},"id":"cd02139d-763e-42ae-a6c1-2f463e7a82fc","type":"PanTool"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(2.256410, 2.461538]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[0.0],"label":["(2.256410, 2.461538]"],"line_alpha":[1.0],"line_color":["black"],"width":[0.2051282051282053],"x":["2.358974"],"y":[0.0]}},"id":"84d62263-dc11-43c8-98e9-8c8759f9aef3","type":"ColumnDataSource"},{"attributes":{"data_source":{"id":"9a78ab60-dc99-4428-ab65-65ebe22ee36b","type":"ColumnDataSource"},"glyph":{"id":"2474ff56-8b9b-4ea9-8b01-7c34ca74d15d","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"e3367e2b-eca5-42ed-810f-195f18343bbb","type":"GlyphRenderer"},{"attributes":{"plot":{"id":"0a4ac7b2-5e5d-4da6-b840-f6945579b6bf","subtype":"Chart","type":"Plot"}},"id":"b6f173a1-9d95-4e6f-93c1-206c160fd5f0","type":"WheelZoomTool"},{"attributes":{"data_source":{"id":"2a6cdb74-4867-403c-8676-8ef0134ca660","type":"ColumnDataSource"},"glyph":{"id":"68105310-8c15-4b9d-80a2-51536a2c823c","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"599b20ee-3a2c-44bb-86fc-2b2ba3d4639a","type":"GlyphRenderer"},{"attributes":{"overlay":{"id":"ef082280-6b7e-4146-98bb-29a1fb3c09e8","type":"BoxAnnotation"},"plot":{"id":"0a4ac7b2-5e5d-4da6-b840-f6945579b6bf","subtype":"Chart","type":"Plot"}},"id":"6cd6c5f6-8dd1-43de-9764-58ceb6e3c7ff","type":"BoxZoomTool"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(6.153846, 6.358974]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[0.0],"label":["(6.153846, 6.358974]"],"line_alpha":[1.0],"line_color":["black"],"width":[0.2051282051282053],"x":["6.25641"],"y":[0.0]}},"id":"9a78ab60-dc99-4428-ab65-65ebe22ee36b","type":"ColumnDataSource"},{"attributes":{"plot":{"id":"0a4ac7b2-5e5d-4da6-b840-f6945579b6bf","subtype":"Chart","type":"Plot"}},"id":"7446428d-50be-4983-bd78-fa3d9864f12e","type":"SaveTool"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"ff843572-5cf3-474d-bdf4-a90a1ec27336","type":"Rect"},{"attributes":{"plot":{"id":"0a4ac7b2-5e5d-4da6-b840-f6945579b6bf","subtype":"Chart","type":"Plot"}},"id":"5ea443c5-a1e5-4c06-9ad7-434097c35175","type":"ResetTool"},{"attributes":{"data_source":{"id":"a8c16704-2ada-418c-a367-2ecdfbeb7c3e","type":"ColumnDataSource"},"glyph":{"id":"9d00c0f7-fa9f-4020-bd1f-7e86955ae3a8","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"32b1eafe-9256-48d0-b7f1-8470a2de533c","type":"GlyphRenderer"},{"attributes":{"plot":{"id":"0a4ac7b2-5e5d-4da6-b840-f6945579b6bf","subtype":"Chart","type":"Plot"}},"id":"3f0e3052-1e18-4762-bdd0-b34cba7bed1b","type":"HelpTool"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(5.333333, 5.538462]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[0.0],"label":["(5.333333, 5.538462]"],"line_alpha":[1.0],"line_color":["black"],"width":[0.2051282051282053],"x":["5.435897499999999"],"y":[0.0]}},"id":"8e1c64fe-2493-40aa-973b-02b8842f9e6c","type":"ColumnDataSource"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(6.358974, 6.564103]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[0.0],"label":["(6.358974, 6.564103]"],"line_alpha":[1.0],"line_color":["black"],"width":[0.2051282051282053],"x":["6.4615385"],"y":[0.0]}},"id":"58bf03da-740d-4e20-a9c9-dcc22c8a7a06","type":"ColumnDataSource"},{"attributes":{"data_source":{"id":"5883f6eb-bdb8-44f6-9d21-45038f7c7021","type":"ColumnDataSource"},"glyph":{"id":"6cc14cca-0504-45ab-916f-361a27eee451","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"3ef5e01c-a294-476f-bc56-7e85c7f399b1","type":"GlyphRenderer"},{"attributes":{"data_source":{"id":"8e1c64fe-2493-40aa-973b-02b8842f9e6c","type":"ColumnDataSource"},"glyph":{"id":"0cf50091-ebea-4390-a8db-a4676b7ef513","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"0c0e3cda-9e62-4aa8-9939-9681d5a410a9","type":"GlyphRenderer"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"a11da88e-c6ab-429a-8842-df93defdf5f1","type":"Rect"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(5.128205, 5.333333]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[0.0],"label":["(5.128205, 5.333333]"],"line_alpha":[1.0],"line_color":["black"],"width":[0.2051282051282053],"x":["5.2307690000000004"],"y":[0.0]}},"id":"a0b9fed0-2031-4c7a-9ad7-b1002fb29d0b","type":"ColumnDataSource"},{"attributes":{"data_source":{"id":"44e161dd-1939-4595-a809-2401256bcff5","type":"ColumnDataSource"},"glyph":{"id":"65b5ce27-6f40-4fae-986a-de42bc84bac7","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"71f19382-28f7-497b-8fa4-d5d857287cc4","type":"GlyphRenderer"},{"attributes":{"data_source":{"id":"58bf03da-740d-4e20-a9c9-dcc22c8a7a06","type":"ColumnDataSource"},"glyph":{"id":"ff843572-5cf3-474d-bdf4-a90a1ec27336","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"a64705ea-b2c8-4de6-93da-812e6444aa29","type":"GlyphRenderer"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"f0e803fe-dc01-43fc-bc3f-c503a5fa0ce2","type":"Rect"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"5978aee2-0e51-4ed4-ae6b-5e19434ffcd5","type":"Rect"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"a9745236-26c0-45fd-87c8-95707f20cf40","type":"Rect"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(1.846154, 2.051282]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[28.0],"label":["(1.846154, 2.051282]"],"line_alpha":[1.0],"line_color":["black"],"width":[0.20512820512820507],"x":["1.948718"],"y":[14.0]}},"id":"a8c16704-2ada-418c-a367-2ecdfbeb7c3e","type":"ColumnDataSource"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(2.051282, 2.256410]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[0.0],"label":["(2.051282, 2.256410]"],"line_alpha":[1.0],"line_color":["black"],"width":[0.2051282051282053],"x":["2.1538459999999997"],"y":[0.0]}},"id":"1cba1185-1a77-4499-aeb3-850b1382e05d","type":"ColumnDataSource"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"65b5ce27-6f40-4fae-986a-de42bc84bac7","type":"Rect"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"316e501f-173c-4275-a846-2b95a6b5b476","type":"Rect"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(3.897436, 4.102564]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[18.0],"label":["(3.897436, 4.102564]"],"line_alpha":[1.0],"line_color":["black"],"width":[0.20512820512820484],"x":["4.0"],"y":[9.0]}},"id":"44e161dd-1939-4595-a809-2401256bcff5","type":"ColumnDataSource"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"fe1f64c4-b3ab-456e-9093-48233539d2ea","type":"Rect"},{"attributes":{"data_source":{"id":"7a889557-bd21-43ee-b12a-604990f03e29","type":"ColumnDataSource"},"glyph":{"id":"f0e803fe-dc01-43fc-bc3f-c503a5fa0ce2","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"b97165b1-f57c-4f6c-bca8-58a61ee7adac","type":"GlyphRenderer"},{"attributes":{"data_source":{"id":"412fb63e-120f-4f82-ad28-f69acf339ca3","type":"ColumnDataSource"},"glyph":{"id":"6d93c4a7-1235-4e23-a248-4941a225b896","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"9558bf13-afe8-4d65-bb33-e797796f7d01","type":"GlyphRenderer"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(1.641026, 1.846154]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[0.0],"label":["(1.641026, 1.846154]"],"line_alpha":[1.0],"line_color":["black"],"width":[0.20512820512820507],"x":["1.7435900000000002"],"y":[0.0]}},"id":"5883f6eb-bdb8-44f6-9d21-45038f7c7021","type":"ColumnDataSource"},{"attributes":{"data_source":{"id":"665eb82b-f50e-4e1c-a017-2a17c0c1df55","type":"ColumnDataSource"},"glyph":{"id":"a9745236-26c0-45fd-87c8-95707f20cf40","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"db56dc85-7f38-408f-a297-153f44e7e5f2","type":"GlyphRenderer"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"9d00c0f7-fa9f-4020-bd1f-7e86955ae3a8","type":"Rect"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"30cde64d-6e3b-4ee9-97bd-dfe27f9165ff","type":"Rect"},{"attributes":{"data_source":{"id":"ff027760-b2f5-404b-babf-759af56c52b8","type":"ColumnDataSource"},"glyph":{"id":"5978aee2-0e51-4ed4-ae6b-5e19434ffcd5","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"bc6ad9a2-b4f3-46e9-b0b6-c30e067bc48e","type":"GlyphRenderer"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(4.102564, 4.307692]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[0.0],"label":["(4.102564, 4.307692]"],"line_alpha":[1.0],"line_color":["black"],"width":[0.2051282051282053],"x":["4.205128"],"y":[0.0]}},"id":"2355a989-ed4c-4a3d-8211-5b4098898954","type":"ColumnDataSource"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"1bf79e1e-bb02-487e-b19b-e5cb2c45f9d4","type":"Rect"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(6.769231, 6.974359]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[0.0],"label":["(6.769231, 6.974359]"],"line_alpha":[1.0],"line_color":["black"],"width":[0.2051282051282053],"x":["6.8717950000000005"],"y":[0.0]}},"id":"665eb82b-f50e-4e1c-a017-2a17c0c1df55","type":"ColumnDataSource"},{"attributes":{"data_source":{"id":"a0b9fed0-2031-4c7a-9ad7-b1002fb29d0b","type":"ColumnDataSource"},"glyph":{"id":"b287cf27-3d1a-48cb-90f7-d7707184f695","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"168d1e47-5666-4f68-9f5c-63f50a12c9bf","type":"GlyphRenderer"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(5.948718, 6.153846]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[0.0],"label":["(5.948718, 6.153846]"],"line_alpha":[1.0],"line_color":["black"],"width":[0.2051282051282044],"x":["6.0512820000000005"],"y":[0.0]}},"id":"ff027760-b2f5-404b-babf-759af56c52b8","type":"ColumnDataSource"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"0cf50091-ebea-4390-a8db-a4676b7ef513","type":"Rect"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"b287cf27-3d1a-48cb-90f7-d7707184f695","type":"Rect"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(6.974359, 7.179487]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[0.0],"label":["(6.974359, 7.179487]"],"line_alpha":[1.0],"line_color":["black"],"width":[0.2051282051282044],"x":["7.076923"],"y":[0.0]}},"id":"0a6dfd77-4471-43b6-9466-6941bca80cd2","type":"ColumnDataSource"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"305092fe-1643-483f-9195-312549e95317","type":"Rect"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"00e231cf-c1a7-4a1c-b8c8-a3ce7d029c79","type":"Rect"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(7.179487, 7.384615]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[0.0],"label":["(7.179487, 7.384615]"],"line_alpha":[1.0],"line_color":["black"],"width":[0.2051282051282053],"x":["7.282051"],"y":[0.0]}},"id":"26095023-0b72-4a03-9ddd-de5dd94b6d3b","type":"ColumnDataSource"},{"attributes":{"data_source":{"id":"5eb83cc7-c988-42dd-8ec0-5d5dd8e9a73a","type":"ColumnDataSource"},"glyph":{"id":"fe1f64c4-b3ab-456e-9093-48233539d2ea","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"517caff9-b39b-4154-8970-8fa6b45fa17e","type":"GlyphRenderer"},{"attributes":{"data_source":{"id":"0a6dfd77-4471-43b6-9466-6941bca80cd2","type":"ColumnDataSource"},"glyph":{"id":"305092fe-1643-483f-9195-312549e95317","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"eb80e16b-6fae-46a5-a511-2364f6848898","type":"GlyphRenderer"},{"attributes":{"axis_label":"SibSp","formatter":{"id":"ad07699f-45d3-4d8d-95b5-76cd150f7e4a","type":"BasicTickFormatter"},"plot":{"id":"0a4ac7b2-5e5d-4da6-b840-f6945579b6bf","subtype":"Chart","type":"Plot"},"ticker":{"id":"f1a5f538-524b-4ee3-9808-cd50b6294a93","type":"BasicTicker"}},"id":"f38a4b37-ceaa-4d70-a661-5ade3222c176","type":"LinearAxis"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"326a2d77-645a-48bf-be8d-e350483855d7","type":"Rect"},{"attributes":{"data_source":{"id":"26095023-0b72-4a03-9ddd-de5dd94b6d3b","type":"ColumnDataSource"},"glyph":{"id":"00e231cf-c1a7-4a1c-b8c8-a3ce7d029c79","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"7cae2a19-83a2-44a9-926e-e4bb48898589","type":"GlyphRenderer"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(7.794872, 8.000000]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[7.0],"label":["(7.794872, 8.000000]"],"line_alpha":[1.0],"line_color":["black"],"width":[0.2051282051282053],"x":["7.897436"],"y":[3.5]}},"id":"c2586ba8-eaa9-4c40-b07b-efa32f681bed","type":"ColumnDataSource"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(7.589744, 7.794872]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[0.0],"label":["(7.589744, 7.794872]"],"line_alpha":[1.0],"line_color":["black"],"width":[0.2051282051282053],"x":["7.692308"],"y":[0.0]}},"id":"5eb83cc7-c988-42dd-8ec0-5d5dd8e9a73a","type":"ColumnDataSource"},{"attributes":{"data_source":{"id":"c2586ba8-eaa9-4c40-b07b-efa32f681bed","type":"ColumnDataSource"},"glyph":{"id":"326a2d77-645a-48bf-be8d-e350483855d7","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"a4e92e21-c4d5-494d-86d6-fd61d2828065","type":"GlyphRenderer"},{"attributes":{"axis_label":"Count( Sibsp )","formatter":{"id":"02aecbcc-50c2-4302-b4df-faf21988fcaf","type":"BasicTickFormatter"},"plot":{"id":"0a4ac7b2-5e5d-4da6-b840-f6945579b6bf","subtype":"Chart","type":"Plot"},"ticker":{"id":"7200b6ff-ee00-4126-90ef-20e83353a1de","type":"BasicTicker"}},"id":"49c493b5-2878-4ba7-86f6-12e0b7ee6563","type":"LinearAxis"},{"attributes":{},"id":"7200b6ff-ee00-4126-90ef-20e83353a1de","type":"BasicTicker"},{"attributes":{"dimension":1,"plot":{"id":"0a4ac7b2-5e5d-4da6-b840-f6945579b6bf","subtype":"Chart","type":"Plot"},"ticker":{"id":"7200b6ff-ee00-4126-90ef-20e83353a1de","type":"BasicTicker"}},"id":"965ec388-bb8f-466f-80b0-14546b28b2dd","type":"Grid"},{"attributes":{},"id":"02aecbcc-50c2-4302-b4df-faf21988fcaf","type":"BasicTickFormatter"},{"attributes":{},"id":"ad07699f-45d3-4d8d-95b5-76cd150f7e4a","type":"BasicTickFormatter"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["[0.000000, 0.205128]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[608.0],"label":["[0.000000, 0.205128]"],"line_alpha":[1.0],"line_color":["black"],"width":[0.20512820512820512],"x":["0.102564"],"y":[304.0]}},"id":"6c0a33c2-2acc-4379-8358-064edce69d79","type":"ColumnDataSource"},{"attributes":{},"id":"f1a5f538-524b-4ee3-9808-cd50b6294a93","type":"BasicTicker"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"6cc14cca-0504-45ab-916f-361a27eee451","type":"Rect"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"41d943f2-7b34-49b6-a29b-e29388346c68","type":"Rect"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"d8a90b91-03e4-4942-82e7-e267e301cbc6","type":"Rect"},{"attributes":{"data_source":{"id":"f06925a6-f3b3-4cc6-8398-88f56116da5e","type":"ColumnDataSource"},"glyph":{"id":"d8a90b91-03e4-4942-82e7-e267e301cbc6","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"9fafb47a-6846-4f81-82c4-c632c735f28b","type":"GlyphRenderer"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"e8b9cabd-910d-4520-8e6a-aed2675c2aeb","type":"Rect"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(0.205128, 0.410256]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[0.0],"label":["(0.205128, 0.410256]"],"line_alpha":[1.0],"line_color":["black"],"width":[0.20512820512820512],"x":["0.307692"],"y":[0.0]}},"id":"f06925a6-f3b3-4cc6-8398-88f56116da5e","type":"ColumnDataSource"},{"attributes":{"data_source":{"id":"6c0a33c2-2acc-4379-8358-064edce69d79","type":"ColumnDataSource"},"glyph":{"id":"41d943f2-7b34-49b6-a29b-e29388346c68","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"51451761-ab2a-41c8-a5d0-e78fa4a85f1a","type":"GlyphRenderer"},{"attributes":{"data_source":{"id":"a484ae0b-70e6-40d4-a202-26081c74a3df","type":"ColumnDataSource"},"glyph":{"id":"0a27b185-426e-4dfb-b16d-95c92a148e69","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"ff9711ab-e06c-4d79-a9a5-4c0b42454805","type":"GlyphRenderer"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"0a27b185-426e-4dfb-b16d-95c92a148e69","type":"Rect"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(7.384615, 7.589744]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[0.0],"label":["(7.384615, 7.589744]"],"line_alpha":[1.0],"line_color":["black"],"width":[0.2051282051282053],"x":["7.4871795"],"y":[0.0]}},"id":"7a889557-bd21-43ee-b12a-604990f03e29","type":"ColumnDataSource"},{"attributes":{"data_source":{"id":"b1f66258-b8da-4e33-a725-80468b0af244","type":"ColumnDataSource"},"glyph":{"id":"9a4ac4c1-799d-4b39-a7c3-1addb959967e","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"e46deb94-9cff-484f-94de-f19c47d9e332","type":"GlyphRenderer"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"51fdd02d-d5e6-48e2-9432-4e010166e13d","type":"Rect"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"ab6dc795-40d5-4331-a89f-e61db59f575b","type":"Rect"},{"attributes":{"data_source":{"id":"e94c896a-51de-4273-ba2a-25d994cd31e3","type":"ColumnDataSource"},"glyph":{"id":"3a3bc753-f7b5-4cc7-be1b-6008ac4eeef7","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"e1a96def-7fa1-46a2-80de-65c076aac7c4","type":"GlyphRenderer"},{"attributes":{"data_source":{"id":"09fc49f8-c9f4-4727-a8f8-4522741132ec","type":"ColumnDataSource"},"glyph":{"id":"465bedac-c33e-4ee0-9b6f-837c058aa261","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"e7ce14fa-6316-413f-b3d9-75e8a73922d0","type":"GlyphRenderer"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(4.512821, 4.717949]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[0.0],"label":["(4.512821, 4.717949]"],"line_alpha":[1.0],"line_color":["black"],"width":[0.2051282051282053],"x":["4.615385"],"y":[0.0]}},"id":"fd1d31ff-e9cf-4bb9-9066-4f7b72841244","type":"ColumnDataSource"},{"attributes":{"data_source":{"id":"3203c888-19fc-49d7-815b-f49d996c40ea","type":"ColumnDataSource"},"glyph":{"id":"6b74bd62-ffc1-480d-84cd-9b90968bd9f2","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"97e5e6f0-aa11-4fb5-9b71-724239010e51","type":"GlyphRenderer"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(3.282051, 3.487179]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[0.0],"label":["(3.282051, 3.487179]"],"line_alpha":[1.0],"line_color":["black"],"width":[0.2051282051282053],"x":["3.384615"],"y":[0.0]}},"id":"e94c896a-51de-4273-ba2a-25d994cd31e3","type":"ColumnDataSource"},{"attributes":{"data_source":{"id":"fd1d31ff-e9cf-4bb9-9066-4f7b72841244","type":"ColumnDataSource"},"glyph":{"id":"51fdd02d-d5e6-48e2-9432-4e010166e13d","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"663af8a1-75d0-49c0-abb8-8541e3ef33bd","type":"GlyphRenderer"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(3.487179, 3.692308]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[0.0],"label":["(3.487179, 3.692308]"],"line_alpha":[1.0],"line_color":["black"],"width":[0.20512820512820484],"x":["3.5897435"],"y":[0.0]}},"id":"09fc49f8-c9f4-4727-a8f8-4522741132ec","type":"ColumnDataSource"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"9a4ac4c1-799d-4b39-a7c3-1addb959967e","type":"Rect"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(3.692308, 3.897436]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[0.0],"label":["(3.692308, 3.897436]"],"line_alpha":[1.0],"line_color":["black"],"width":[0.2051282051282053],"x":["3.794872"],"y":[0.0]}},"id":"b1f66258-b8da-4e33-a725-80468b0af244","type":"ColumnDataSource"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"465bedac-c33e-4ee0-9b6f-837c058aa261","type":"Rect"},{"attributes":{"data_source":{"id":"a11923c3-2639-4d31-8e67-7bf75b8a226c","type":"ColumnDataSource"},"glyph":{"id":"84ab2116-d11e-4f9b-b6b2-89bc9789d10b","type":"Rect"},"hover_glyph":null,"muted_glyph":null},"id":"3cfeccb9-d68e-4c9e-a6f2-215fadc9b62b","type":"GlyphRenderer"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"84ab2116-d11e-4f9b-b6b2-89bc9789d10b","type":"Rect"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(2.666667, 2.871795]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[0.0],"label":["(2.666667, 2.871795]"],"line_alpha":[1.0],"line_color":["black"],"width":[0.2051282051282053],"x":["2.769231"],"y":[0.0]}},"id":"a11923c3-2639-4d31-8e67-7bf75b8a226c","type":"ColumnDataSource"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(3.076923, 3.282051]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[0.0],"label":["(3.076923, 3.282051]"],"line_alpha":[1.0],"line_color":["black"],"width":[0.2051282051282053],"x":["3.179487"],"y":[0.0]}},"id":"466cff73-b0f2-43a9-b9a7-a8dfe6397719","type":"ColumnDataSource"},{"attributes":{"fill_alpha":{"field":"fill_alpha"},"fill_color":{"field":"color"},"height":{"field":"height","units":"data"},"line_color":{"field":"line_color"},"width":{"field":"width","units":"data"},"x":{"field":"x"},"y":{"field":"y"}},"id":"6b74bd62-ffc1-480d-84cd-9b90968bd9f2","type":"Rect"},{"attributes":{"callback":null,"column_names":["x","y","width","height","color","fill_alpha","line_color","line_alpha","label"],"data":{"chart_index":["(2.871795, 3.076923]"],"color":["#f22c40"],"fill_alpha":[0.8],"height":[16.0],"label":["(2.871795, 3.076923]"],"line_alpha":[1.0],"line_color":["black"],"width":[0.20512820512820484],"x":["2.9743589999999998"],"y":[8.0]}},"id":"3203c888-19fc-49d7-815b-f49d996c40ea","type":"ColumnDataSource"}],"root_ids":["0a4ac7b2-5e5d-4da6-b840-f6945579b6bf"]},"title":"Bokeh Application","version":"0.12.5"}};
            var render_items = [{"docid":"1c624a9d-6b1b-44a0-9e68-7e10bb22623c","elementid":"ae0cf855-51c2-406e-a829-bd14a146ba08","modelid":"0a4ac7b2-5e5d-4da6-b840-f6945579b6bf"}];
            
            Bokeh.embed.embed_items(docs_json, render_items);
          };
          if (document.readyState != "loading") fn();
          else document.addEventListener("DOMContentLoaded", fn);
        })();
      },
      function(Bokeh) {
      }
    ];
  
    function run_inline_js() {
      
      if ((window.Bokeh !== undefined) || (force === true)) {
        for (var i = 0; i < inline_js.length; i++) {
          inline_js[i](window.Bokeh);
        }if (force === true) {
          display_loaded();
        }} else if (Date.now() < window._bokeh_timeout) {
        setTimeout(run_inline_js, 100);
      } else if (!window._bokeh_failed_load) {
        console.log("Bokeh: BokehJS failed to load within specified timeout.");
        window._bokeh_failed_load = true;
      } else if (force !== true) {
        var cell = $(document.getElementById("ae0cf855-51c2-406e-a829-bd14a146ba08")).parents('.cell').data().cell;
        cell.output_area.append_execute_result(NB_LOAD_WARNING)
      }
  
    }
  
    if (window._bokeh_is_loading === 0) {
      console.log("Bokeh: BokehJS loaded, going straight to plotting");
      run_inline_js();
    } else {
      load_libs(js_urls, function() {
        console.log("Bokeh: BokehJS plotting callback run at", now());
        run_inline_js();
      });
    }
  }(this));
</script>



```python
from bokeh.charts import Donut, show
pie_chart = Donut(train_df["Sex"])
show(pie_chart)
```




    <div class="bk-root">
        <div class="bk-plotdiv" id="7c374016-e2bd-4b28-9cb0-f6675a4dee7d"></div>
    </div>
<script type="text/javascript">
  
  (function(global) {
    function now() {
      return new Date();
    }
  
    var force = false;
  
    if (typeof (window._bokeh_onload_callbacks) === "undefined" || force === true) {
      window._bokeh_onload_callbacks = [];
      window._bokeh_is_loading = undefined;
    }
  
  
    
    if (typeof (window._bokeh_timeout) === "undefined" || force === true) {
      window._bokeh_timeout = Date.now() + 0;
      window._bokeh_failed_load = false;
    }
  
    var NB_LOAD_WARNING = {'data': {'text/html':
       "<div style='background-color: #fdd'>\n"+
       "<p>\n"+
       "BokehJS does not appear to have successfully loaded. If loading BokehJS from CDN, this \n"+
       "may be due to a slow or bad network connection. Possible fixes:\n"+
       "</p>\n"+
       "<ul>\n"+
       "<li>re-rerun `output_notebook()` to attempt to load from CDN again, or</li>\n"+
       "<li>use INLINE resources instead, as so:</li>\n"+
       "</ul>\n"+
       "<code>\n"+
       "from bokeh.resources import INLINE\n"+
       "output_notebook(resources=INLINE)\n"+
       "</code>\n"+
       "</div>"}};
  
    function display_loaded() {
      if (window.Bokeh !== undefined) {
        var el = document.getElementById("7c374016-e2bd-4b28-9cb0-f6675a4dee7d");
        el.textContent = "BokehJS " + Bokeh.version + " successfully loaded.";
      } else if (Date.now() < window._bokeh_timeout) {
        setTimeout(display_loaded, 100)
      }
    }
  
    function run_callbacks() {
      window._bokeh_onload_callbacks.forEach(function(callback) { callback() });
      delete window._bokeh_onload_callbacks
      console.info("Bokeh: all callbacks have finished");
    }
  
    function load_libs(js_urls, callback) {
      window._bokeh_onload_callbacks.push(callback);
      if (window._bokeh_is_loading > 0) {
        console.log("Bokeh: BokehJS is being loaded, scheduling callback at", now());
        return null;
      }
      if (js_urls == null || js_urls.length === 0) {
        run_callbacks();
        return null;
      }
      console.log("Bokeh: BokehJS not loaded, scheduling load and callback at", now());
      window._bokeh_is_loading = js_urls.length;
      for (var i = 0; i < js_urls.length; i++) {
        var url = js_urls[i];
        var s = document.createElement('script');
        s.src = url;
        s.async = false;
        s.onreadystatechange = s.onload = function() {
          window._bokeh_is_loading--;
          if (window._bokeh_is_loading === 0) {
            console.log("Bokeh: all BokehJS libraries loaded");
            run_callbacks()
          }
        };
        s.onerror = function() {
          console.warn("failed to load library " + url);
        };
        console.log("Bokeh: injecting script tag for BokehJS library: ", url);
        document.getElementsByTagName("head")[0].appendChild(s);
      }
    };var element = document.getElementById("7c374016-e2bd-4b28-9cb0-f6675a4dee7d");
    if (element == null) {
      console.log("Bokeh: ERROR: autoload.js configured with elementid '7c374016-e2bd-4b28-9cb0-f6675a4dee7d' but no matching script tag was found. ")
      return false;
    }
  
    var js_urls = [];
  
    var inline_js = [
      function(Bokeh) {
        (function() {
          var fn = function() {
            var docs_json = {"430b66ee-7fb9-4933-819a-22a11a5e9022":{"roots":{"references":[{"attributes":{"callback":null,"column_names":["text","x","y","text_angle"],"data":{"text":["female","male"],"text_angle":{"__ndarray__":"AAAAAAAAAAAAAAAAAAAAAA==","dtype":"float64","shape":[2]},"x":[0.5031259267658803,-0.5031259267658804],"y":[1.0062252738904813,-1.006225273890481]}},"id":"36bb9654-be97-4447-9700-2f5f6a08c0d0","type":"ColumnDataSource"},{"attributes":{"axis_label":null,"formatter":{"id":"7d950ea4-cfe0-4647-911a-40a250cd3b6a","type":"BasicTickFormatter"},"plot":{"id":"354990be-6b5c-435f-8061-f26a0ad708ff","subtype":"Chart","type":"Plot"},"ticker":{"id":"5bc0da82-abbd-46da-8ae2-7ebda3e00ff8","type":"BasicTicker"},"visible":false},"id":"72d3fd3a-71d2-402f-aa7c-88ce3cc99d9f","type":"LinearAxis"},{"attributes":{"callback":null,"plot":{"id":"354990be-6b5c-435f-8061-f26a0ad708ff","subtype":"Chart","type":"Plot"},"tooltips":[["Value","@values"]]},"id":"aadd287e-bd97-4bad-9858-964900587e0a","type":"HoverTool"},{"attributes":{"axis_label":null,"formatter":{"id":"7389a2a7-fcc1-444e-bfbb-f593c8004ed3","type":"BasicTickFormatter"},"plot":{"id":"354990be-6b5c-435f-8061-f26a0ad708ff","subtype":"Chart","type":"Plot"},"ticker":{"id":"b7513866-2a9d-49a3-a2a2-e4266b2016e0","type":"BasicTicker"},"visible":false},"id":"cc1a6564-6cdf-456d-b6b5-9aa58ce9c3e6","type":"LinearAxis"},{"attributes":{"plot":{"id":"354990be-6b5c-435f-8061-f26a0ad708ff","subtype":"Chart","type":"Plot"}},"id":"b9cbd8a5-7f4b-45cd-ba61-f44eb8a3ec35","type":"WheelZoomTool"},{"attributes":{},"id":"7d950ea4-cfe0-4647-911a-40a250cd3b6a","type":"BasicTickFormatter"},{"attributes":{"end_angle":{"field":"end","units":"rad"},"fill_alpha":{"value":0.8},"fill_color":{"field":"color"},"inner_radius":{"field":"inners","units":"data"},"line_color":{"value":"White"},"outer_radius":{"field":"outers","units":"data"},"start_angle":{"field":"start","units":"rad"},"x":{"value":0},"y":{"value":0}},"id":"014edc7c-3dab-45b9-98ad-5dd621553498","type":"AnnularWedge"},{"attributes":{"callback":null,"end":1.6500000000000001,"start":-1.6500000000000001},"id":"fc798885-ed51-43a4-9630-78de3cd7ffc8","type":"Range1d"},{"attributes":{"callback":null,"end":1.6500000000000001,"start":-1.6500000000000001},"id":"43ea3a08-9fad-4c25-a498-02786a34c48a","type":"Range1d"},{"attributes":{},"id":"7389a2a7-fcc1-444e-bfbb-f593c8004ed3","type":"BasicTickFormatter"},{"attributes":{"plot":{"id":"354990be-6b5c-435f-8061-f26a0ad708ff","subtype":"Chart","type":"Plot"}},"id":"90157534-e266-4fe1-80fd-32ea7546a06b","type":"PanTool"},{"attributes":{"data_source":{"id":"36bb9654-be97-4447-9700-2f5f6a08c0d0","type":"ColumnDataSource"},"glyph":{"id":"8be9171f-e0c9-4c11-a7a4-16a4a9f8a8cd","type":"Text"},"hover_glyph":null,"muted_glyph":null},"id":"abf3a60a-1b24-4ad1-b836-ba741ed4f1d3","type":"GlyphRenderer"},{"attributes":{"angle":{"field":"text_angle","units":"rad"},"text_align":"center","text_baseline":"middle","text_font_size":{"value":"10pt"},"x":{"field":"x"},"y":{"field":"y"}},"id":"8be9171f-e0c9-4c11-a7a4-16a4a9f8a8cd","type":"Text"},{"attributes":{"data_source":{"id":"3e6897b5-a60c-461d-9249-9177aead71ee","type":"ColumnDataSource"},"glyph":{"id":"014edc7c-3dab-45b9-98ad-5dd621553498","type":"AnnularWedge"},"hover_glyph":null,"muted_glyph":null},"id":"4137a8a6-bb03-43aa-892a-99f7737a44da","type":"GlyphRenderer"},{"attributes":{},"id":"5bc0da82-abbd-46da-8ae2-7ebda3e00ff8","type":"BasicTicker"},{"attributes":{},"id":"b7513866-2a9d-49a3-a2a2-e4266b2016e0","type":"BasicTicker"},{"attributes":{"overlay":{"id":"2ed4de7e-bf49-4726-a5bd-b764ee3d6541","type":"BoxAnnotation"},"plot":{"id":"354990be-6b5c-435f-8061-f26a0ad708ff","subtype":"Chart","type":"Plot"}},"id":"489a3c3a-a807-4d6c-8266-b83386d5d816","type":"BoxZoomTool"},{"attributes":{"bottom_units":"screen","fill_alpha":{"value":0.5},"fill_color":{"value":"lightgrey"},"left_units":"screen","level":"overlay","line_alpha":{"value":1.0},"line_color":{"value":"black"},"line_dash":[4,4],"line_width":{"value":2},"plot":null,"render_mode":"css","right_units":"screen","top_units":"screen"},"id":"2ed4de7e-bf49-4726-a5bd-b764ee3d6541","type":"BoxAnnotation"},{"attributes":{"below":[{"id":"72d3fd3a-71d2-402f-aa7c-88ce3cc99d9f","type":"LinearAxis"}],"css_classes":null,"left":[{"id":"cc1a6564-6cdf-456d-b6b5-9aa58ce9c3e6","type":"LinearAxis"}],"plot_height":400,"plot_width":400,"renderers":[{"id":"2ed4de7e-bf49-4726-a5bd-b764ee3d6541","type":"BoxAnnotation"},{"id":"4137a8a6-bb03-43aa-892a-99f7737a44da","type":"GlyphRenderer"},{"id":"abf3a60a-1b24-4ad1-b836-ba741ed4f1d3","type":"GlyphRenderer"},{"id":"983c89e5-86e7-44f0-a82e-8ab18fc559ad","type":"Legend"},{"id":"72d3fd3a-71d2-402f-aa7c-88ce3cc99d9f","type":"LinearAxis"},{"id":"cc1a6564-6cdf-456d-b6b5-9aa58ce9c3e6","type":"LinearAxis"}],"title":{"id":"753aa118-9a0e-4ef9-9dba-f04055a797d5","type":"Title"},"tool_events":{"id":"81f1de8d-cf90-4036-b884-23d9001bb2fa","type":"ToolEvents"},"toolbar":{"id":"6cdadfe1-b3f1-463a-866a-f1cd67f18cdb","type":"Toolbar"},"x_mapper_type":"auto","x_range":{"id":"43ea3a08-9fad-4c25-a498-02786a34c48a","type":"Range1d"},"y_mapper_type":"auto","y_range":{"id":"fc798885-ed51-43a4-9630-78de3cd7ffc8","type":"Range1d"}},"id":"354990be-6b5c-435f-8061-f26a0ad708ff","subtype":"Chart","type":"Plot"},{"attributes":{},"id":"81f1de8d-cf90-4036-b884-23d9001bb2fa","type":"ToolEvents"},{"attributes":{"plot":{"id":"354990be-6b5c-435f-8061-f26a0ad708ff","subtype":"Chart","type":"Plot"}},"id":"0d164994-7ccb-499f-acbb-83816aed917d","type":"SaveTool"},{"attributes":{"location":"top_left","plot":{"id":"354990be-6b5c-435f-8061-f26a0ad708ff","subtype":"Chart","type":"Plot"}},"id":"983c89e5-86e7-44f0-a82e-8ab18fc559ad","type":"Legend"},{"attributes":{"plot":null,"text":null},"id":"753aa118-9a0e-4ef9-9dba-f04055a797d5","type":"Title"},{"attributes":{"plot":{"id":"354990be-6b5c-435f-8061-f26a0ad708ff","subtype":"Chart","type":"Plot"}},"id":"e0be0d6a-d665-4051-9d9d-ff11ebcfce92","type":"HelpTool"},{"attributes":{"active_drag":"auto","active_scroll":"auto","active_tap":"auto","tools":[{"id":"90157534-e266-4fe1-80fd-32ea7546a06b","type":"PanTool"},{"id":"b9cbd8a5-7f4b-45cd-ba61-f44eb8a3ec35","type":"WheelZoomTool"},{"id":"489a3c3a-a807-4d6c-8266-b83386d5d816","type":"BoxZoomTool"},{"id":"0d164994-7ccb-499f-acbb-83816aed917d","type":"SaveTool"},{"id":"85e103fd-44f4-4617-9464-a78b7da8f187","type":"ResetTool"},{"id":"e0be0d6a-d665-4051-9d9d-ff11ebcfce92","type":"HelpTool"},{"id":"aadd287e-bd97-4bad-9858-964900587e0a","type":"HoverTool"}]},"id":"6cdadfe1-b3f1-463a-866a-f1cd67f18cdb","type":"Toolbar"},{"attributes":{"plot":{"id":"354990be-6b5c-435f-8061-f26a0ad708ff","subtype":"Chart","type":"Plot"}},"id":"85e103fd-44f4-4617-9464-a78b7da8f187","type":"ResetTool"},{"attributes":{"callback":null,"column_names":["end","level","start","values","inners","outers","centers","color","Sex"],"data":{"Sex":["female","male"],"centers":{"__ndarray__":"AAAAAAAA8j8AAAAAAADyPw==","dtype":"float64","shape":[2]},"color":["#f22c40","#5ab738"],"end":{"__ndarray__":"MBugfta2AUAZLURU+yEZQA==","dtype":"float64","shape":[2]},"inners":{"__ndarray__":"AAAAAAAAAAAAAAAAAAAAAA==","dtype":"float64","shape":[2]},"level":{"__ndarray__":"AAAAAAAAAAAAAAAAAAAAAA==","dtype":"float64","shape":[2]},"outers":{"__ndarray__":"AAAAAAAA+D8AAAAAAAD4Pw==","dtype":"float64","shape":[2]},"start":{"__ndarray__":"AAAAAAAAAAAwG6B+1rYBQA==","dtype":"float64","shape":[2]},"values":[314,577]}},"id":"3e6897b5-a60c-461d-9249-9177aead71ee","type":"ColumnDataSource"}],"root_ids":["354990be-6b5c-435f-8061-f26a0ad708ff"]},"title":"Bokeh Application","version":"0.12.5"}};
            var render_items = [{"docid":"430b66ee-7fb9-4933-819a-22a11a5e9022","elementid":"7c374016-e2bd-4b28-9cb0-f6675a4dee7d","modelid":"354990be-6b5c-435f-8061-f26a0ad708ff"}];
            
            Bokeh.embed.embed_items(docs_json, render_items);
          };
          if (document.readyState != "loading") fn();
          else document.addEventListener("DOMContentLoaded", fn);
        })();
      },
      function(Bokeh) {
      }
    ];
  
    function run_inline_js() {
      
      if ((window.Bokeh !== undefined) || (force === true)) {
        for (var i = 0; i < inline_js.length; i++) {
          inline_js[i](window.Bokeh);
        }if (force === true) {
          display_loaded();
        }} else if (Date.now() < window._bokeh_timeout) {
        setTimeout(run_inline_js, 100);
      } else if (!window._bokeh_failed_load) {
        console.log("Bokeh: BokehJS failed to load within specified timeout.");
        window._bokeh_failed_load = true;
      } else if (force !== true) {
        var cell = $(document.getElementById("7c374016-e2bd-4b28-9cb0-f6675a4dee7d")).parents('.cell').data().cell;
        cell.output_area.append_execute_result(NB_LOAD_WARNING)
      }
  
    }
  
    if (window._bokeh_is_loading === 0) {
      console.log("Bokeh: BokehJS loaded, going straight to plotting");
      run_inline_js();
    } else {
      load_libs(js_urls, function() {
        console.log("Bokeh: BokehJS plotting callback run at", now());
        run_inline_js();
      });
    }
  }(this));
</script>



```python
from bokeh.charts import Donut, show
pie_chart = Donut(train_df["Embarked"])
show(pie_chart)
```




    <div class="bk-root">
        <div class="bk-plotdiv" id="75c9071a-3d6b-488a-b2b8-b87d9d55f8f2"></div>
    </div>
<script type="text/javascript">
  
  (function(global) {
    function now() {
      return new Date();
    }
  
    var force = false;
  
    if (typeof (window._bokeh_onload_callbacks) === "undefined" || force === true) {
      window._bokeh_onload_callbacks = [];
      window._bokeh_is_loading = undefined;
    }
  
  
    
    if (typeof (window._bokeh_timeout) === "undefined" || force === true) {
      window._bokeh_timeout = Date.now() + 0;
      window._bokeh_failed_load = false;
    }
  
    var NB_LOAD_WARNING = {'data': {'text/html':
       "<div style='background-color: #fdd'>\n"+
       "<p>\n"+
       "BokehJS does not appear to have successfully loaded. If loading BokehJS from CDN, this \n"+
       "may be due to a slow or bad network connection. Possible fixes:\n"+
       "</p>\n"+
       "<ul>\n"+
       "<li>re-rerun `output_notebook()` to attempt to load from CDN again, or</li>\n"+
       "<li>use INLINE resources instead, as so:</li>\n"+
       "</ul>\n"+
       "<code>\n"+
       "from bokeh.resources import INLINE\n"+
       "output_notebook(resources=INLINE)\n"+
       "</code>\n"+
       "</div>"}};
  
    function display_loaded() {
      if (window.Bokeh !== undefined) {
        var el = document.getElementById("75c9071a-3d6b-488a-b2b8-b87d9d55f8f2");
        el.textContent = "BokehJS " + Bokeh.version + " successfully loaded.";
      } else if (Date.now() < window._bokeh_timeout) {
        setTimeout(display_loaded, 100)
      }
    }
  
    function run_callbacks() {
      window._bokeh_onload_callbacks.forEach(function(callback) { callback() });
      delete window._bokeh_onload_callbacks
      console.info("Bokeh: all callbacks have finished");
    }
  
    function load_libs(js_urls, callback) {
      window._bokeh_onload_callbacks.push(callback);
      if (window._bokeh_is_loading > 0) {
        console.log("Bokeh: BokehJS is being loaded, scheduling callback at", now());
        return null;
      }
      if (js_urls == null || js_urls.length === 0) {
        run_callbacks();
        return null;
      }
      console.log("Bokeh: BokehJS not loaded, scheduling load and callback at", now());
      window._bokeh_is_loading = js_urls.length;
      for (var i = 0; i < js_urls.length; i++) {
        var url = js_urls[i];
        var s = document.createElement('script');
        s.src = url;
        s.async = false;
        s.onreadystatechange = s.onload = function() {
          window._bokeh_is_loading--;
          if (window._bokeh_is_loading === 0) {
            console.log("Bokeh: all BokehJS libraries loaded");
            run_callbacks()
          }
        };
        s.onerror = function() {
          console.warn("failed to load library " + url);
        };
        console.log("Bokeh: injecting script tag for BokehJS library: ", url);
        document.getElementsByTagName("head")[0].appendChild(s);
      }
    };var element = document.getElementById("75c9071a-3d6b-488a-b2b8-b87d9d55f8f2");
    if (element == null) {
      console.log("Bokeh: ERROR: autoload.js configured with elementid '75c9071a-3d6b-488a-b2b8-b87d9d55f8f2' but no matching script tag was found. ")
      return false;
    }
  
    var js_urls = [];
  
    var inline_js = [
      function(Bokeh) {
        (function() {
          var fn = function() {
            var docs_json = {"6c031431-e72b-4203-8abf-34d9bb067efa":{"roots":{"references":[{"attributes":{"callback":null,"plot":{"id":"a7b16242-cb1f-4142-aea1-60d67ae92524","subtype":"Chart","type":"Plot"},"tooltips":[["Value","@values"]]},"id":"c4dcd97e-a6aa-418a-9244-9722d5045518","type":"HoverTool"},{"attributes":{"location":"top_left","plot":{"id":"a7b16242-cb1f-4142-aea1-60d67ae92524","subtype":"Chart","type":"Plot"}},"id":"22fa81c1-d788-4e1b-a92b-b65601e71bd6","type":"Legend"},{"attributes":{"below":[{"id":"25880e2f-b2f2-460a-9ee0-c05836361d81","type":"LinearAxis"}],"css_classes":null,"left":[{"id":"06fa5939-f86c-467d-b31c-35fb26397633","type":"LinearAxis"}],"plot_height":400,"plot_width":400,"renderers":[{"id":"150c7564-d203-471f-b304-5f1042ddb503","type":"BoxAnnotation"},{"id":"09454010-7010-48a6-ab74-e22d906b60f3","type":"GlyphRenderer"},{"id":"ee455e4d-1f22-4d1e-98c3-7fadf4a62386","type":"GlyphRenderer"},{"id":"22fa81c1-d788-4e1b-a92b-b65601e71bd6","type":"Legend"},{"id":"25880e2f-b2f2-460a-9ee0-c05836361d81","type":"LinearAxis"},{"id":"06fa5939-f86c-467d-b31c-35fb26397633","type":"LinearAxis"}],"title":{"id":"09564e0f-7225-4e54-9d1e-e2f898337466","type":"Title"},"tool_events":{"id":"56724ea9-faad-4f09-baeb-b56a0d2cb65f","type":"ToolEvents"},"toolbar":{"id":"93eb0fe0-fccc-44fe-8298-cfd8096715a2","type":"Toolbar"},"x_mapper_type":"auto","x_range":{"id":"8aa10103-3274-4604-a672-5bf583a709b2","type":"Range1d"},"y_mapper_type":"auto","y_range":{"id":"6d891c3b-6db5-4d23-934e-4bab58aa57e9","type":"Range1d"}},"id":"a7b16242-cb1f-4142-aea1-60d67ae92524","subtype":"Chart","type":"Plot"},{"attributes":{"end_angle":{"field":"end","units":"rad"},"fill_alpha":{"value":0.8},"fill_color":{"field":"color"},"inner_radius":{"field":"inners","units":"data"},"line_color":{"value":"White"},"outer_radius":{"field":"outers","units":"data"},"start_angle":{"field":"start","units":"rad"},"x":{"value":0},"y":{"value":0}},"id":"5f451310-a442-4430-9ae6-11c86228ae6e","type":"AnnularWedge"},{"attributes":{"angle":{"field":"text_angle","units":"rad"},"text_align":"center","text_baseline":"middle","text_font_size":{"value":"10pt"},"x":{"field":"x"},"y":{"field":"y"}},"id":"32dbe84d-e114-4f48-861b-0aae5c7259c4","type":"Text"},{"attributes":{"callback":null,"end":1.6500000000000001,"start":-1.6500000000000001},"id":"8aa10103-3274-4604-a672-5bf583a709b2","type":"Range1d"},{"attributes":{},"id":"20944b55-6332-4339-be56-9fff172d3d24","type":"BasicTicker"},{"attributes":{},"id":"45a579f3-b6e6-48ed-89f2-8fa99c247d60","type":"BasicTicker"},{"attributes":{"data_source":{"id":"e3596ac5-3baa-4479-9b64-f98eb84942c3","type":"ColumnDataSource"},"glyph":{"id":"5f451310-a442-4430-9ae6-11c86228ae6e","type":"AnnularWedge"},"hover_glyph":null,"muted_glyph":null},"id":"09454010-7010-48a6-ab74-e22d906b60f3","type":"GlyphRenderer"},{"attributes":{},"id":"b824a612-22a6-4df5-aa6e-aa4eb1173860","type":"BasicTickFormatter"},{"attributes":{},"id":"3dd014ec-8e06-4b5e-9120-492058ed028b","type":"BasicTickFormatter"},{"attributes":{"plot":{"id":"a7b16242-cb1f-4142-aea1-60d67ae92524","subtype":"Chart","type":"Plot"}},"id":"59e86a07-0c75-45b5-8144-05b48900c10a","type":"SaveTool"},{"attributes":{"callback":null,"end":1.6500000000000001,"start":-1.6500000000000001},"id":"6d891c3b-6db5-4d23-934e-4bab58aa57e9","type":"Range1d"},{"attributes":{},"id":"56724ea9-faad-4f09-baeb-b56a0d2cb65f","type":"ToolEvents"},{"attributes":{"active_drag":"auto","active_scroll":"auto","active_tap":"auto","tools":[{"id":"e6f54e7c-3c3f-4524-b288-5877cddf60ac","type":"PanTool"},{"id":"b99cf37e-13dc-4299-9d12-e6e2b7db1ad1","type":"WheelZoomTool"},{"id":"d44b48ff-c86a-4a2d-af67-96e7de8c1079","type":"BoxZoomTool"},{"id":"59e86a07-0c75-45b5-8144-05b48900c10a","type":"SaveTool"},{"id":"514a4b71-c457-4b15-90e8-5020d69deefe","type":"ResetTool"},{"id":"40c620ae-07fa-46ee-bb81-48d249b75b9d","type":"HelpTool"},{"id":"c4dcd97e-a6aa-418a-9244-9722d5045518","type":"HoverTool"}]},"id":"93eb0fe0-fccc-44fe-8298-cfd8096715a2","type":"Toolbar"},{"attributes":{"callback":null,"column_names":["end","level","start","values","inners","outers","centers","color","Embarked"],"data":{"Embarked":["C","Q","S"],"centers":{"__ndarray__":"AAAAAAAA8j8AAAAAAADyPwAAAAAAAPI/","dtype":"float64","shape":[3]},"color":["#f22c40","#5ab738","#407ee7"],"end":{"__ndarray__":"k2Afdnv/8j+BF0MMlLT7PxgtRFT7IRlA","dtype":"float64","shape":[3]},"inners":{"__ndarray__":"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA","dtype":"float64","shape":[3]},"level":{"__ndarray__":"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA","dtype":"float64","shape":[3]},"outers":{"__ndarray__":"AAAAAAAA+D8AAAAAAAD4PwAAAAAAAPg/","dtype":"float64","shape":[3]},"start":{"__ndarray__":"AAAAAAAAAACTYB92e//yP4EXQwyUtPs/","dtype":"float64","shape":[3]},"values":[168,77,644]}},"id":"e3596ac5-3baa-4479-9b64-f98eb84942c3","type":"ColumnDataSource"},{"attributes":{"bottom_units":"screen","fill_alpha":{"value":0.5},"fill_color":{"value":"lightgrey"},"left_units":"screen","level":"overlay","line_alpha":{"value":1.0},"line_color":{"value":"black"},"line_dash":[4,4],"line_width":{"value":2},"plot":null,"render_mode":"css","right_units":"screen","top_units":"screen"},"id":"150c7564-d203-471f-b304-5f1042ddb503","type":"BoxAnnotation"},{"attributes":{"data_source":{"id":"0e143fdc-47fe-46cd-8587-cdd1f5e2fdfd","type":"ColumnDataSource"},"glyph":{"id":"32dbe84d-e114-4f48-861b-0aae5c7259c4","type":"Text"},"hover_glyph":null,"muted_glyph":null},"id":"ee455e4d-1f22-4d1e-98c3-7fadf4a62386","type":"GlyphRenderer"},{"attributes":{"plot":{"id":"a7b16242-cb1f-4142-aea1-60d67ae92524","subtype":"Chart","type":"Plot"}},"id":"e6f54e7c-3c3f-4524-b288-5877cddf60ac","type":"PanTool"},{"attributes":{"plot":{"id":"a7b16242-cb1f-4142-aea1-60d67ae92524","subtype":"Chart","type":"Plot"}},"id":"40c620ae-07fa-46ee-bb81-48d249b75b9d","type":"HelpTool"},{"attributes":{"axis_label":null,"formatter":{"id":"3dd014ec-8e06-4b5e-9120-492058ed028b","type":"BasicTickFormatter"},"plot":{"id":"a7b16242-cb1f-4142-aea1-60d67ae92524","subtype":"Chart","type":"Plot"},"ticker":{"id":"45a579f3-b6e6-48ed-89f2-8fa99c247d60","type":"BasicTicker"},"visible":false},"id":"25880e2f-b2f2-460a-9ee0-c05836361d81","type":"LinearAxis"},{"attributes":{"plot":{"id":"a7b16242-cb1f-4142-aea1-60d67ae92524","subtype":"Chart","type":"Plot"}},"id":"b99cf37e-13dc-4299-9d12-e6e2b7db1ad1","type":"WheelZoomTool"},{"attributes":{"overlay":{"id":"150c7564-d203-471f-b304-5f1042ddb503","type":"BoxAnnotation"},"plot":{"id":"a7b16242-cb1f-4142-aea1-60d67ae92524","subtype":"Chart","type":"Plot"}},"id":"d44b48ff-c86a-4a2d-af67-96e7de8c1079","type":"BoxZoomTool"},{"attributes":{"callback":null,"column_names":["text","x","y","text_angle"],"data":{"text":["C","Q","S"],"text_angle":{"__ndarray__":"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA","dtype":"float64","shape":[3]},"x":[0.9324943248109319,0.12497234067990526,-0.7290406936143061],"y":[0.6293483409014473,1.1180370808094808,-0.8568107533489361]}},"id":"0e143fdc-47fe-46cd-8587-cdd1f5e2fdfd","type":"ColumnDataSource"},{"attributes":{"plot":null,"text":null},"id":"09564e0f-7225-4e54-9d1e-e2f898337466","type":"Title"},{"attributes":{"plot":{"id":"a7b16242-cb1f-4142-aea1-60d67ae92524","subtype":"Chart","type":"Plot"}},"id":"514a4b71-c457-4b15-90e8-5020d69deefe","type":"ResetTool"},{"attributes":{"axis_label":null,"formatter":{"id":"b824a612-22a6-4df5-aa6e-aa4eb1173860","type":"BasicTickFormatter"},"plot":{"id":"a7b16242-cb1f-4142-aea1-60d67ae92524","subtype":"Chart","type":"Plot"},"ticker":{"id":"20944b55-6332-4339-be56-9fff172d3d24","type":"BasicTicker"},"visible":false},"id":"06fa5939-f86c-467d-b31c-35fb26397633","type":"LinearAxis"}],"root_ids":["a7b16242-cb1f-4142-aea1-60d67ae92524"]},"title":"Bokeh Application","version":"0.12.5"}};
            var render_items = [{"docid":"6c031431-e72b-4203-8abf-34d9bb067efa","elementid":"75c9071a-3d6b-488a-b2b8-b87d9d55f8f2","modelid":"a7b16242-cb1f-4142-aea1-60d67ae92524"}];
            
            Bokeh.embed.embed_items(docs_json, render_items);
          };
          if (document.readyState != "loading") fn();
          else document.addEventListener("DOMContentLoaded", fn);
        })();
      },
      function(Bokeh) {
      }
    ];
  
    function run_inline_js() {
      
      if ((window.Bokeh !== undefined) || (force === true)) {
        for (var i = 0; i < inline_js.length; i++) {
          inline_js[i](window.Bokeh);
        }if (force === true) {
          display_loaded();
        }} else if (Date.now() < window._bokeh_timeout) {
        setTimeout(run_inline_js, 100);
      } else if (!window._bokeh_failed_load) {
        console.log("Bokeh: BokehJS failed to load within specified timeout.");
        window._bokeh_failed_load = true;
      } else if (force !== true) {
        var cell = $(document.getElementById("75c9071a-3d6b-488a-b2b8-b87d9d55f8f2")).parents('.cell').data().cell;
        cell.output_area.append_execute_result(NB_LOAD_WARNING)
      }
  
    }
  
    if (window._bokeh_is_loading === 0) {
      console.log("Bokeh: BokehJS loaded, going straight to plotting");
      run_inline_js();
    } else {
      load_libs(js_urls, function() {
        console.log("Bokeh: BokehJS plotting callback run at", now());
        run_inline_js();
      });
    }
  }(this));
</script>



```python
df = pd.melt(df, id_vars=['abbr'],
             value_vars=['bronze', 'silver', 'gold'],
             value_name='medal_count', var_name='medal')
```


```python
df = pd.melt(train_df[["Embarked","Sex"]], id_vars=['Sex'],
             value_vars=['male', 'female'],
             value_name='count', var_name='sex')
```


```python
from bokeh.charts import Donut, show
pie_chart = Donut(df,values='count')
show(pie_chart)
```


    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    <ipython-input-108-3250e1b7d091> in <module>()
          1 from bokeh.charts import Donut, show
    ----> 2 pie_chart = Donut(df,values='count')
          3 show(pie_chart)
    

    C:\Program Files\Anaconda3\lib\site-packages\bokeh\charts\builders\donut_builder.py in Donut(data, label, values, color, agg, hover_tool, hover_text, plot_height, plot_width, xgrid, ygrid, **kw)
        114         kw['agg'] = agg
        115 
    --> 116     chart = create_and_build(DonutBuilder, data, **kw)
        117 
        118     chart.left[0].visible = False
    

    C:\Program Files\Anaconda3\lib\site-packages\bokeh\charts\builder.py in create_and_build(builder_class, *data, **kws)
         54     chart_kws = {k: v for k, v in kws.items() if k not in builder_props}
         55     chart = Chart(**chart_kws)
    ---> 56     chart.add_builder(builder)
         57     chart.start_plot()
         58 
    

    C:\Program Files\Anaconda3\lib\site-packages\bokeh\charts\chart.py in add_builder(self, builder)
        151     def add_builder(self, builder):
        152         self._builders.append(builder)
    --> 153         builder.create(self)
        154 
        155     def add_ranges(self, dim, range):
    

    C:\Program Files\Anaconda3\lib\site-packages\bokeh\charts\builder.py in create(self, chart)
        504         # call methods that allow customized setup by subclasses
        505         self.setup()
    --> 506         self.process_data()
        507 
        508         # create and add renderers to chart
    

    C:\Program Files\Anaconda3\lib\site-packages\bokeh\charts\builders\donut_builder.py in process_data(self)
        214                                         agg=self.agg,
        215                                         level_width=self.level_width,
    --> 216                                         level_spacing=self.level_spacing)
        217 
        218         # add placeholder color column that will be assigned colors
    

    C:\Program Files\Anaconda3\lib\site-packages\bokeh\charts\utils.py in build_wedge_source(df, cat_cols, agg_col, agg, level_width, level_spacing)
        345 def build_wedge_source(df, cat_cols, agg_col=None, agg='mean', level_width=0.5,
        346                        level_spacing=0.01):
    --> 347     df = cat_to_polar(df, cat_cols, agg_col, agg, level_width)
        348 
        349     add_wedge_spacing(df, level_spacing)
    

    C:\Program Files\Anaconda3\lib\site-packages\bokeh\charts\utils.py in cat_to_polar(df, cat_cols, agg_col, agg, level_width)
        409 
        410         if agg_col is not None and agg is not None:
    --> 411             gb = getattr(getattr(df.groupby(level_cols), agg_col), agg)()
        412         else:
        413             cols = [col for col in df.columns if col != 'index']
    

    AttributeError: 'function' object has no attribute 'sum'



```python
trainDF.describe()
```


```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
 
trainDF = pd.read_csv(open("C:/Users/Jérémie/IdeaProjects/titanic-kaggle-py/data/train.csv", 'r'))
 
rt = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=3, random_state=0)
 
columns = ["Fare", "Pclass","SibSp","Parch"]
 
labels = trainDF["Survived"].values
features = trainDF[list(columns)].values
 
cross_val_result = cross_val_score(rt, features, labels, cv=5, n_jobs=-1)
score_mean = cross_val_result.mean()
score_max = cross_val_result.max()

print("{0} -> RF mean: {1}".format(columns, score_mean))
print("{0} -> RF_max: {1}".format(columns, score_max))

```

    ['Fare', 'Pclass', 'SibSp', 'Parch'] -> RF mean: 0.7038571353084471
    ['Fare', 'Pclass', 'SibSp', 'Parch'] -> RF_max: 0.7696629213483146
    


```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=0)
```


```python
from sklearn import svm

clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
clf.score(X_test, y_test)   
```




    0.69402985074626866


