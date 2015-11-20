import java.util.ArrayList;

public class Factory {
	public static void main(String[] args) {
		String login = "abc";
		String password = "12345";
		ArrayList<FormValidator> validators = new ArrayList<>();

		validators.add(FormValidator.create(FormValidator.Types.LOGIN, login));
		validators.add(FormValidator.create(FormValidator.Types.PASSWORD, password));

		String errors = "";

		for (int i = 0; i < validators.size(); ++i) {
			if (!validators.get(i).validate()) {
				errors = errors + validators.get(i).getErrorMsg() + '\n';
			}
		}
		
		if (errors.length() == 0) {
			System.out.println("No errors");
		} else {
			System.out.println("Errors:");
			System.out.print(errors);
		}
	}
}
