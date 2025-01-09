// Selectors
const questionContainer = document.getElementById("question-container");
const addQuestionButton = document.querySelector(".add-question");
const exportButton = document.querySelector(".export");

// Add a new question
let questionCount = 1;
addQuestionButton.addEventListener("click", () => {
  questionCount++;
  const newQuestion = document.createElement("div");
  newQuestion.classList.add("question");
  newQuestion.innerHTML = `
    <label>${questionCount < 10 ? "0" + questionCount : questionCount}. </label>
    <textarea rows="2" placeholder="Type here..."></textarea>
    <button class="add-subpart">Add Subpart</button>
  `;
  questionContainer.appendChild(newQuestion);
});

// Add subparts to a question
questionContainer.addEventListener("click", (event) => {
  if (event.target.classList.contains("add-subpart")) {
    const subpart = document.createElement("textarea");
    subpart.rows = 2;
    subpart.placeholder = "Type subpart here...";
    subpart.style.marginTop = "5px";
    event.target.parentElement.appendChild(subpart);
  }
});

// Export to XLSX
exportButton.addEventListener("click", () => {
  const allQuestions = document.querySelectorAll(".question");
  const data = [];
  
  allQuestions.forEach((question, index) => {
    const questionText = question.querySelector("textarea").value;
    
    // Collect subparts
    const subparts = [...question.querySelectorAll("textarea")].slice(1).map(sp => sp.value);
    const answerText = [questionText, ...subparts].join(" ");  // Combine question and subparts as a single answer
    
    // Set the marks column as "Not Assigned" or leave it blank
    const marks = "Not Assigned"; // Or you can leave it as empty string, depending on preference
    
    // Push data in the structure [Question Number, Answer Text, Marks]
    data.push([`Q${index + 1}`, answerText, marks]);
  });

  // Create worksheet and workbook
  const ws = XLSX.utils.aoa_to_sheet([
    ["Question Number", "Answer", "Marks"],  // Header Row
    ...data  // Data rows
  ]);
  
  const wb = XLSX.utils.book_new();
  XLSX.utils.book_append_sheet(wb, ws, "Questions");

  // Write the XLSX file
  XLSX.writeFile(wb, "questions.xlsx");
});





