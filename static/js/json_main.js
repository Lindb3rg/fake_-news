function displayFileName() {
    const fileInput = document.getElementById('fileInput');
    const fileNameDisplay = document.getElementById('fileNameDisplay');

    if (fileInput.files.length > 0) {
      fileNameDisplay.textContent = `Selected File: ${fileInput.files[0].name}`;
    } else {
      fileNameDisplay.textContent = '';
    }
  }