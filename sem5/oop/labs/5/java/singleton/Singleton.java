public class Singleton {
	public static void main(String[] args) {
		Connection con = Connection.getInstance();
		
		con.action();
	}
}
