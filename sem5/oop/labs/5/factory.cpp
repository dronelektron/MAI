/*
Пример применения паттерна проектирования Factory в проекте.

Можно использовать Factory для валидаторов пользовательских форм
*/

#include <iostream>
#include <string>
#include <vector>

class FormValidator
{
public:
	enum Types
	{
		LOGIN = 0,
		PASSWORD
	};

	virtual bool validate() = 0;
	virtual std::string getErrorMsg() = 0;

	static FormValidator* create(Types type, std::string value);

protected:
	std::string m_value;
};

class LoginValidator : public FormValidator
{
public:
	LoginValidator(std::string value);
	
	bool validate();
	std::string getErrorMsg();
};

class PasswordValidator : public FormValidator
{
public:
	PasswordValidator(std::string value);
	
	bool validate();
	std::string getErrorMsg();
};

int main()
{
	std::string login = "abc";
	std::string password = "12345";
	std::vector<FormValidator*> validators;

	validators.push_back(FormValidator::create(FormValidator::LOGIN, login));
	validators.push_back(FormValidator::create(FormValidator::PASSWORD, password));

	std::string errors = "";

	for (int i = 0; i < validators.size(); ++i)
		if (!validators[i]->validate())
			errors += validators[i]->getErrorMsg() + '\n';

	if (errors.length() == 0)
		std::cout << "No errors" << std::endl;
	else
		std::cout << "Errors:" << std::endl << errors;
	
	return 0;
}

FormValidator* FormValidator::create(Types type, std::string value)
{
	FormValidator* result = NULL;

	switch (type)
	{
		case LOGIN:
		{
			result = new LoginValidator(value);

			break;
		}

		case PASSWORD:
		{
			result = new PasswordValidator(value);

			break;
		}
	}

	return result;
}

LoginValidator::LoginValidator(std::string value)
{
	m_value = value;
}

bool LoginValidator::validate()
{
	return m_value.length() >= 3;
}

std::string LoginValidator::getErrorMsg()
{
	return "Length of login must be greater than 2";
}

PasswordValidator::PasswordValidator(std::string value)
{
	m_value = value;
}

bool PasswordValidator::validate()
{
	return m_value.length() >= 8;
}

std::string PasswordValidator::getErrorMsg()
{
	return "Length of password must be greater than 7";
}
