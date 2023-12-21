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

// function toggleTextForm() {
//   fetch('/text')
//       .then(response => response.text())
//       .then(data => {
//           document.getElementById('contentContainer').innerHTML = data;
//       })
//       .catch(error => {
//           console.error('Error:', error);
//       });
// }

// function loadContentFromIndexRoute() {
//   fetch('/')
//       .then(response => response.text())
//       .then(data => {
//           document.getElementById('contentContainer').innerHTML = data;
//       })
//       .catch(error => {
//           console.error('Error:', error);
//       });
// }

// function switchToIndexContent() {
//   loadContentFromIndexRoute();
// }




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






// function toggleTextForm() {
//   var form = document.getElementById('textForm');
//   var customDiv = document.querySelector('.custom-div');
//   var submitButton = document.querySelector('.submit-button');
//   if (form.style.display === 'none') {
//     form.style.display = 'block';
//     customDiv.style.height = 'auto'; // Let the custom div adjust to the form content
//     submitButton.style.backgroundColor = '#25455f';
//   } else {
//     form.style.display = 'none';
//     customDiv.style.height = 'auto'; // Set custom div height when form is hidden
//     submitButton.style.backgroundColor = '';
//   }
// }

// function toggleUseTextForm() {
//   const csrfToken = document.querySelector('input[name="csrf_token"]').value;
//   fetch('/toggle_use_text_form', {
//       method: 'POST',
//       headers: {
//         'Content-Type': 'application/json',
//         'X-CSRFToken': csrfToken
//     }
//   })
//   .then(response => {
//       if (response.ok) {
//         window.location.reload();
//           // Toggle was successful
//           // You can perform additional actions here if needed
//       } else {
//           // Handle errors if needed
//       }
//   })
//   .catch(error => {
//       // Handle network errors
//   });
// }



//   function toggleUseTextForm() {
//     fetch('/?text=True', {
//         method: 'POST'
//     })
//     .then(response => {
//         if (response.ok) {
//             window.location.reload(); // Reload the page to display updated content
//         } else {
//             console.error('Failed to toggle the form.');
//         }
//     })
//     .catch(error => {
//         console.error('Error toggling the form:', error);
//     });
// }




// function toggleFileForm() {
//   var form = document.getElementById('fileForm');
//   var customDiv = document.querySelector('.custom-div');
//   var uploadButton = document.querySelector('.upload-button');
//   if (form.style.display === 'none') {
//     form.style.display = 'block';
//     customDiv.style.height = 'auto'; // Let the custom div adjust to the form content
//     uploadButton.style.backgroundColor = '#25455f';
//   } else {
//     form.style.display = 'none';
//     customDiv.style.height = 'auto'; // Set custom div height when form is hidden
//     uploadButton.style.backgroundColor = '';
//   }
// }


// function toggleUseTextForm() {
//   var xhr = new XMLHttpRequest();
//   xhr.open('POST', '/', true);
//   xhr.setRequestHeader('Content-Type', 'application/x-www-form-urlencoded');
//   xhr.onreadystatechange = function() {
//       if (xhr.readyState === XMLHttpRequest.DONE) {
//           if (xhr.status === 200) {
//               // Request was successful, re-render the form
//               location.reload(); // Refresh the page
//           } else {
//               // Handle errors if needed
//               console.error('Error:', xhr.statusText);
//           }
//       }
//   };
//   xhr.send('toggle_text=true'); // Send toggle parameter as 'true'
// }



  // function toggleUseTextForm() {

  //     fetch('/?method=text_submit', {
  //         method: 'POST',
  //         headers: {
  //             'Content-Type': 'application/json',
  //         },
  //         body: JSON.stringify({ toggle: true }), 

  //     })
  //     .then(response => {
  //         if (response.ok) {
  //           toggleTextForm()
  //         } else {
  //             // Handle errors if needed
  //         }
  //     })
  //     .catch(error => {
  //         // Handle network errors
  //     });
  // }





