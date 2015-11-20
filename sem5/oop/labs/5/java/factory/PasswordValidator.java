public class PasswordValidator extends FormValidator {
	public PasswordValidator(String value) {
		this.value = value;
	}

	@Override
	public boolean validate() {
		return value.length() > 7;
	}

	@Override
	public String getErrorMsg() {
		return "Length of password must be greater than 7";
	}
}
