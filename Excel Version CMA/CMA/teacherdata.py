import pandas as pd


def create_sample_files():
    """Create sample Excel files for testing."""

    # Teacher's answer
    teacher_answer = """The water cycle, also known as the hydrologic cycle, is the continuous movement of water on, above, and below the surface of the Earth. It involves processes like evaporation, condensation, precipitation, and runoff. Solar energy drives the cycle, causing water to evaporate from oceans and land surfaces. This water vapor rises, cools, and condenses to form clouds, eventually falling as precipitation."""

    # Different types of student answers for testing
    student_answers = {
        'good': """The water cycle is the continuous movement of water on Earth. It includes evaporation from oceans and land, condensation in clouds, precipitation as rain or snow, and runoff back to oceans. Solar energy powers this cycle, making water evaporate and later condense in clouds.""",

        'too_short': """The water cycle is how water moves around Earth. Water evaporates, forms clouds, and falls as rain.""",

        'off_topic': """Water is a chemical compound made up of hydrogen and oxygen atoms. It can exist in three states: solid, liquid, and gas."""
    }

    # Create teacher's file
    pd.DataFrame({'Answer': [teacher_answer]}).to_excel('teacher_answers.xlsx', index=False)

    # Create student's file (using 'good' answer by default)
    pd.DataFrame({'Answer': [student_answers['good']]}).to_excel('student_answers.xlsx', index=False)

    # Create additional test files
    pd.DataFrame({'Answer': [student_answers['too_short']]}).to_excel('student_answers_short.xlsx', index=False)
    pd.DataFrame({'Answer': [student_answers['off_topic']]}).to_excel('student_answers_offtopic.xlsx', index=False)


if __name__ == "__main__":
    create_sample_files()
    print("Sample files created successfully!")
    print(
        "Files created: teacher_answers.xlsx, student_answers.xlsx, student_answers_short.xlsx, student_answers_offtopic.xlsx")