function displayFileName() {
  console.log("File input changed");
  var input = document.getElementById('fileInput');
  var display = document.getElementById('fileNameDisplay');

  if (input.files.length > 0) {
      display.textContent = input.files[0].name;
  } else {
      display.textContent = "No file selected";
  }
}

let isTextContentDisplayed = false;


function toggleTextForm() {
    if (isTextContentDisplayed) {
        clearContentContainer();
        isTextContentDisplayed = false;
    } else {
        fetch('/text')
            .then(response => response.text())
            .then(data => {
                document.getElementById('contentContainer').innerHTML = data;
                isTextContentDisplayed = true;
            })
            .catch(error => {
                console.error('Error:', error);
            });
    }
}

function toggleFileForm() {
  if (isTextContentDisplayed) {
      clearContentContainer();
      isTextContentDisplayed = false;
  } else {
      fetch('/file')
          .then(response => response.text())
          .then(data => {
              document.getElementById('contentContainer').innerHTML = data;
              isTextContentDisplayed = true;
          })
          .catch(error => {
              console.error('Error:', error);
          });
  }
}

function clearContentContainer() {
  document.getElementById('contentContainer').innerHTML = '';
}
