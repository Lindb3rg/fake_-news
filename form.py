from flask_wtf import FlaskForm
from wtforms import StringField, RadioField, SubmitField, validators
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms.validators import DataRequired, ValidationError, Length, EqualTo, NumberRange, InputRequired



class textForm(FlaskForm):
    text = StringField("Text here", validators=[DataRequired(), Length(min=4, max=200)])
    model = RadioField('model', choices=[
        ('sequential', 'Sequential Model'),
        ('logistic', 'Logistic Model'),
        ('svm', 'SVM Model'),
        ('all_models', 'All Models'),
    ], validators=[validators.Optional()], default='svm')
    submit = SubmitField("Click To Predict")




class FileForm(FlaskForm):
    css_file = FileField(
        'Upload CSS File',
        validators=[
            FileRequired(),
            FileAllowed(['css'], 'Only CSS files are allowed.')
        ]
    )

    model = RadioField('model', choices=[
        ('sequential', 'Sequential Model'),
        ('logistic', 'Logistic Model'),
        ('svm', 'SVM Model'),
        ('all_models', 'All Models'),
    ], validators=[validators.Optional()], default='svm')

    submit = SubmitField("Click To Predict")
