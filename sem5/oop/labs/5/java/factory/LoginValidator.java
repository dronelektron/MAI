public class LoginValidator extends FormValidator {
	public LoginValidator(String value) {
		this.value = value;
	}

	@Override
	public boolean validate() {
		return value.length() > 2;
	}

	@Override
	public String getErrorMsg() {
		return "Length of login must be greater than 2";
	}
}
