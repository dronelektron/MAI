public abstract class FormValidator {
	protected String value;
	
	public enum Types {
		LOGIN,
		PASSWORD
	}

	public abstract boolean validate();
	public abstract String getErrorMsg();

	public static FormValidator create(Types type, String value) {
		FormValidator result = null;

		switch (type) {
			case LOGIN:
				result = new LoginValidator(value);

				break;

			case PASSWORD:
				result = new PasswordValidator(value);

				break;
		}

		return result;
	}
}
