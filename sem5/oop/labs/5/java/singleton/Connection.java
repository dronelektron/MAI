public class Connection {
	private Connection() {
		this.host = "localhost";
		this.port = 1234;
	}

	public void action() {
		System.out.println("Connection action");
	}

	public static Connection getInstance() {
		return instance;
	}

	private String host;
	private int port;

	private static Connection instance = new Connection();
}
