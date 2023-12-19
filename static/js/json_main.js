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



