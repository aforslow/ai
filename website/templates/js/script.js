var ws;
// var ws1;

function onLoad() {
    ws = new WebSocket("ws://31.208.78.186:5891/websocket");

    //image
    ws.onmessage = function(json_string) {
        var parsed_json = JSON.parse(json_string.data);
        forEach(parsed_json, write_to_field)
  };
}

function write_to_field(key, val) {
    if (document.getElementById(key)) {
      if (key == 'img') {
        document.getElementById(key).src = "data:image/jpg;base64," + val
      } else if (key.startsWith('Q')) {
        document.getElementById(key).innerHTML = val
        var tmp_val = 20*(1+parseFloat(val));
        // var tmp_val2 = 10*tmp_val;
        document.getElementById(key + "Box").style.height = tmp_val.toString() + 'px'
        // document.getElementById(key + "Box").innerHTML = val + 'px'
      } else {
        document.getElementById(key).innerHTML = val
      }
    }
}

function forEach(dict, f) {
  for (key in dict) {
    if (dict.hasOwnProperty(key)) {
      f(key, dict[key])
    }
  }
}

var rendered = false;

function image(src, id) {
    var img = document.createElement("img");
    img.src = src;
    img.id = id
    if(!rendered) {
      document.getElementById('imageDiv').appendChild(img);
      rendered = true;
    } else {
      document.getElementById(id).src = src;
    }
}

function sendMsg() {
    ws.send(document.getElementById('msg').value);
}
function closeMsg() {
  ws.close()
}

function writeSection() {
  var msg = document.getElementById('msg').value;
  var div = document.createElement('div');
  div.setAttribute('class', 'gameDiv');
  div.innerHTML = msg;
  document.getElementById('myMainDiv').appendChild(div);
}

var imageStreamClosed = true;
function toggleImageStream() {
  imageStreamClosed = !imageStreamClosed;
  if (imageStreamClosed) {
    ws.send('closeImageStream');
  } else {
    ws.send('openImageStream');
  }
}
