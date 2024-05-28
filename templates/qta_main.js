const dropArea = document.querySelector("drop-area");
const inputFile = document.querySelector("input-file");

function uploadFile(){

}

dropArea.addEventListener("dragover", (e) => {
    e.preventDefault();
});

dropArea.addEventListener("drop", (e) => {
    e.preventDefault();
    inputFile.files = e.dataTransfer.files;
    uploadFile();
});

