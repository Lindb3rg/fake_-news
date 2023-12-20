function displayFileName() {
  console.log("File input changed");  // Add this line
  var input = document.getElementById('fileInput');
  var display = document.getElementById('fileNameDisplay');

  if (input.files.length > 0) {
      display.textContent = input.files[0].name;
  } else {
      display.textContent = "No file selected";
  }
}

function toggleTextForm() {
  var form = document.getElementById('textForm');
  var customDiv = document.querySelector('.custom-div');
  var submitButton = document.querySelector('.submit-button');
  if (form.style.display === 'none') {
    form.style.display = 'block';
    customDiv.style.height = 'auto'; // Let the custom div adjust to the form content
    submitButton.style.backgroundColor = '#25455f';
  } else {
    form.style.display = 'none';
    customDiv.style.height = 'auto'; // Set custom div height when form is hidden
    submitButton.style.backgroundColor = '';
  }
}

function toggleFileForm() {
  var form = document.getElementById('fileForm');
  var customDiv = document.querySelector('.custom-div');
  var uploadButton = document.querySelector('.upload-button');
  if (form.style.display === 'none') {
    form.style.display = 'block';
    customDiv.style.height = 'auto'; // Let the custom div adjust to the form content
    uploadButton.style.backgroundColor = '#25455f';
  } else {
    form.style.display = 'none';
    customDiv.style.height = 'auto'; // Set custom div height when form is hidden
    uploadButton.style.backgroundColor = '';
  }
}








