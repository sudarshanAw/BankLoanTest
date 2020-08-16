from django import forms

class CustomerForm(forms.Form):

    CREDIT_HISTORY_CHOICES = [
        ('0',0),
        ('1',1),
        ('2',2),
        ('3',3)
    ]
    GENDER_CHOICES = [
        ('Male','Male'),
        ('Female','Female')
    ]

    YES_NO_CHOICES = [
        ('Yes','Yes'),
        ('No','No')
    ]
    EDUCATION_CHOICES = [
        ('Graduate', 'Graduate'),
        ('Not Graduate','Not Graduate')
    ]
    PROPERTY_AREA_CHOICES = [
        ('Rural','Rural'),
        ('Semiurban','Semiurban'),
        ('Urban','Urban')
    ]

    First_Name =  forms.CharField(max_length=50 , widget= forms.TextInput(attrs={'placeholder' : 'Enter your first name'}))
    Last_Name = forms.CharField(max_length=50, widget= forms.TextInput(attrs= {'placeholder' : 'Enter your last name '}))
    Dependents = forms.IntegerField(widget= forms.NumberInput(attrs={'placeholder' : 'Enter the number of dependants'}))
    ApplicantIncome = forms.IntegerField(widget= forms.NumberInput(attrs={'placeholder' : 'Enter your income'}))
    CoapplicantIncome = forms.IntegerField(widget= forms.NumberInput(attrs={'placeholder' : 'Enter Co-applicants income'}))
    LoanAmount = forms.IntegerField(widget=forms.NumberInput(attrs= {'placeholder' : 'Enter the required loan'}))
    Loan_Amount_Term = forms.IntegerField(widget=forms.NumberInput(attrs={'placeholder': 'Enter the loan terms in months'}))
    Credit_History = forms.ChoiceField(choices=CREDIT_HISTORY_CHOICES)
    Gender = forms.ChoiceField(choices=GENDER_CHOICES)
    Married = forms.ChoiceField(choices=YES_NO_CHOICES)
    Education = forms.ChoiceField(choices=EDUCATION_CHOICES)
    Self_Employed = forms.ChoiceField(choices=YES_NO_CHOICES)
    Property_Area = forms.ChoiceField(choices=PROPERTY_AREA_CHOICES)