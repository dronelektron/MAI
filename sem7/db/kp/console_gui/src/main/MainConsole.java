package main;

public class MainConsole {
	public static void main(String[] args) {
		SQLManager manager = new SQLManager();
		String query;

		manager.connect();

		System.out.println("==== ИСХОДНЫЕ ДАННЫЕ ====");

		manager.select("select * from USER_INFO order by USER_INFO_ID");

		System.out.println("==== ВСТАВКА ====");

		query = "insert into USER_INFO (USER_INFO_ID, LOGIN, PASSWORD, FIRST_NAME, SECOND_NAME, EMAIL, PHONE)";
		query += " values (USER_INFO_SEQ.NEXTVAL, 'Dima', '1122', 'Дима', 'Некрасов', 'dimanek@mail.ru', '+79151234567')";

		manager.execute(query);
		manager.select("select * from USER_INFO order by USER_INFO_ID");

		System.out.println("==== ОБНОВЛЕНИЕ ====");

		query = "update USER_INFO ui set ui.PASSWORD = ? where ui.login = ?";

		manager.execute(query, "2143", "Dima");
		manager.select("select * from USER_INFO order by USER_INFO_ID");

		System.out.println("==== УДАЛЕНИЕ ====");

		query = "delete from USER_INFO ui where ui.LOGIN = ?";

		manager.execute(query, "Dima");
		manager.select("select * from USER_INFO order by USER_INFO_ID");

		System.out.println("==== ХРАНИМАЯ ПРОЦЕДУРА ====");

		manager.callProc(1); // storage_id = 1
		manager.callProc(2);

		manager.disconnect();
	}
}
