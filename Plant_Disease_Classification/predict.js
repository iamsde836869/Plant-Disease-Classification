$("#image-selector").change(function () {
    let reader = new FileReader();
    reader.onload = function () {
        let dataURL = reader.result;
        $('#selected-image').attr("src", dataURL);
        $("#prediction-list").empty();
    } 
    let file = $("#image-selector").prop('files')[0];
    reader.readAsDataURL(file);
})

// let model;
// (async function() {
//     model = await tf.loadLayersModel('model/model.json');
//     $('.progress-bar').hide();
// })();

let model;
$( document ).ready(async function () {
	$('.progress-bar').show();
    console.log( "Loading model..." );
    model = await tf.loadLayersModel('http://192.168.43.6:2000/models/model.json');
    console.log( "Model loaded." );
	$('.progress-bar').hide();
});

$("#predict-button").click(async function () {
    let image = $('#selected-image').get(0);
    let tensor = tf.browser.fromPixels(image)
        .resizeNearestNeighbor([128, 128])
        .toFloat()
        .div(tf.scalar(255.0))
        .expandDims();

        let predictions = await model.predict(tensor).data();
        let top5 = Array.from(predictions)
            .map(function (p, i) {
                return {
                    probability: p,
                    className: CLASSES[i] 
                };
            }).sort(function (a, b){
                return b.probability - a.probability;
            }).slice(0, 5);

        $("#prediction-list").empty();
        top5.forEach(function (p) {
            $('#prediction-list').append(`<li>${p.className}: ${p.probability.toFixed(6)}</li>`);
        });
});


