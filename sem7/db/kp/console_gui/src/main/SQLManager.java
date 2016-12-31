package main;

import java.sql.*;
import java.util.List;

public class SQLManager {
	public void connect() {
		try {
			connection = DriverManager.getConnection("jdbc:oracle:thin:@localhost:1521:orcl", "Andy", "1234");
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	public void disconnect() {
		try {
			connection.close();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	public void select(String query, Object... params) {
		ResultSet rs = null;
		PreparedStatement ps = null;

		try {
			ps = connection.prepareStatement(query);

			for (int i = 0; i < params.length; ++i) {
				Object obj = params[i];
				Class cls = obj.getClass();

				if (cls == String.class) {
					ps.setString(i + 1, obj.toString());
				} else if (cls == Integer.class) {
					ps.setInt(i + 1, Integer.valueOf(obj.toString()));
				} else {
					throw new Exception("Unknown type of SQL parameter");
				}
			}

			rs = ps.executeQuery();

			while (rs.next()) {
				ResultSetMetaData rsmd = rs.getMetaData();
				String str = "";

				for (int i = 0; i < rsmd.getColumnCount(); ++i) {
					str += rs.getString(i + 1);

					if (i + 1 < rsmd.getColumnCount()) {
						str += '\t';
					}
				}

				if (output == null) {
					System.out.println(str);
				} else {
					output.add(str);
				}
			}
		} catch (Exception e) {
			e.printStackTrace();
		} finally {
			try {
				if (rs != null) {
					rs.close();
				}

				if (ps != null) {
					ps.close();
				}
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
	}

	public void execute(String query, Object... params) {
		PreparedStatement ps = null;

		try {
			ps = connection.prepareStatement(query);

			for (int i = 0; i < params.length; ++i) {
				Object obj = params[i];
				Class cls = obj.getClass();

				if (cls == String.class) {
					ps.setString(i + 1, obj.toString());
				} else if (cls == Integer.class) {
					ps.setInt(i + 1, Integer.valueOf(obj.toString()));
				} else {
					throw new Exception("Unknown type of SQL parameter");
				}
			}

			ps.execute();
		} catch (Exception e) {
			e.printStackTrace();
		} finally {
			try {
				if (ps != null) {
					ps.close();
				}
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
	}

	public void callProc(int storageId) {
		CallableStatement cs = null;

		try {
			cs = connection.prepareCall("call P_CALC_FREE_SPACE(?, ?)");
			cs.setInt(1, storageId);
			cs.registerOutParameter(2, Types.INTEGER);
			cs.execute();

			System.out.println("Free space for storage " + storageId + ": " + cs.getInt(2) + " bytes");
		} catch (Exception e) {
			e.printStackTrace();
		} finally {
			try {
				if (cs != null) {
					cs.close();
				}
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
	}

	public void setOutput(List<String> output) {
		this.output = output;
	}

	static {
		try {
			Class.forName("oracle.jdbc.OracleDriver");
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	private Connection connection;
	private List<String> output;
}
