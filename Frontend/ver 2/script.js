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
    <textarea rows="3" placeholder="Enter Marks"></textarea>
    <button class="add-subpart">Add Subpart</button>
  `;
  questionContainer.appendChild(newQuestion);
});

// Add subparts to a question
questionContainer.addEventListener("click", (event) => {
  if (event.target.classList.contains("add-subpart")) {
    const questionDiv = event.target.parentElement;
    const subparts = questionDiv.querySelectorAll(".subpart");
    const subpartCount = subparts.length + 1;
    const subpartLabel = String.fromCharCode(65 + subpartCount); // A, B, C...

    const subpartContainer = document.createElement("div");
    subpartContainer.classList.add("subpart");
    subpartContainer.innerHTML = `
      <label>${questionDiv.querySelector("label").textContent.trim()}${subpartLabel}. </label>
      <textarea rows="2" placeholder="Type subpart here..."></textarea>
      <textarea rows="2" placeholder="Enter Marks"></textarea>
    `;
    questionDiv.appendChild(subpartContainer);
  }
});

// Export to XLSX
exportButton.addEventListener("click", () => {
  const allQuestions = document.querySelectorAll(".question");
  const data = [];

  allQuestions.forEach((question, index) => {
    const questionText = question.querySelector("textarea").value;
    const questionMarks = question.querySelector("textarea:nth-of-type(2)").value || "Not Assigned";

    // Add the main question
    data.push([`Q${index + 1}`, questionText, questionMarks]);

    // Collect subparts
    const subparts = question.querySelectorAll(".subpart");
    subparts.forEach((subpart) => {
      const subpartText = subpart.querySelector("textarea").value;
      const subpartMarks = subpart.querySelector("textarea:nth-of-type(2)").value || "Not Assigned";
      const subpartLabel = subpart.querySelector("label").textContent.trim();

      data.push([subpartLabel, subpartText, subpartMarks]);
    });
  });

  // Create worksheet and workbook
  const ws = XLSX.utils.aoa_to_sheet([
    ["Question/Subpart", "Answer", "Marks"], // Header Row
    ...data // Data rows
  ]);

  const wb = XLSX.utils.book_new();
  XLSX.utils.book_append_sheet(wb, ws, "Questions");

  // Write the XLSX file
  XLSX.writeFile(wb, "questions.xlsx");
});
