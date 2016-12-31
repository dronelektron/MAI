package main.rows;

public class UserRow {
	public UserRow(int id, String login, String password, String firstName, String secondName, String email, String phone) {
		this.id = id;
		this.login = login;
		this.password = password;
		this.firstName = firstName;
		this.secondName = secondName;
		this.email = email;
		this.phone = phone;
	}

	public int getId() {
		return id;
	}

	public String getLogin() {
		return login;
	}

	public String getPassword() {
		return password;
	}

	public String getFirstName() {
		return firstName;
	}

	public String getSecondName() {
		return secondName;
	}

	public String getEmail() {
		return email;
	}

	public String getPhone() {
		return phone;
	}

	private int id;
	private String login;
	private String password;
	private String firstName;
	private String secondName;
	private String email;
	private String phone;
}
