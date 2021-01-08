let model;
const webcam = new Webcam(document.getElementById('wc'));
let isPredicting = false;

async function loadMobilenet() {
  const model = await tf.loadLayersModel('Model/model.json');
  return tf.model({inputs: model.inputs, outputs: model.output});
}

async function predict() {
  while (isPredicting) {
    const predictedClass = tf.tidy(() => {
      const img = webcam.capture();
      const predictions = model.predict(img);
      return predictions.as1D().argMax();
    });
    const classId = (await predictedClass.data())[0];
    var predictionText = "";
    switch(classId){
		case 0:
			predictionText = "I see mask";
			break;
		case 1:
			predictionText = "I can't see mask";
			break;
	}
	document.getElementById("prediction").innerText = predictionText;
			
    
    predictedClass.dispose();
    await tf.nextFrame();
  }
}


function doTraining(){
	train();
}

function startPredicting(){
	isPredicting = true;
	predict();
}

function stopPredicting(){
	isPredicting = false;
	predict();
}

async function init(){
	await webcam.setup();
	model = await loadMobilenet();
		
}



init();
