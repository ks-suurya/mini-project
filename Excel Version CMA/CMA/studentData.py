import pandas as pd

# Create a sample student answer
student_answer = "The water cycle is how water moves around Earth. Water evaporates from oceans and lakes, forms clouds through condensation, falls as rain or snow, and then flows back to oceans through rivers."

# Create a DataFrame with the answer
df = pd.DataFrame({'Answer': [student_answer]})

# Save to Excel file
df.to_excel('student_answers.xlsx', index=False)