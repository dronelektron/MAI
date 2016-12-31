package main;

import java.net.URL;
import java.util.ArrayList;
import java.util.List;
import java.util.ResourceBundle;

import javafx.fxml.FXML;
import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
import javafx.fxml.Initializable;
import javafx.scene.control.ComboBox;
import javafx.scene.control.TableColumn;
import javafx.scene.control.TableView;
import javafx.scene.control.TextField;
import javafx.scene.control.cell.PropertyValueFactory;

import main.rows.ShareRow;
import main.rows.UserRow;

public class MainGUIController implements Initializable {
	@Override
	public void initialize(URL location, ResourceBundle resources) {
		t1c1.setCellValueFactory(new PropertyValueFactory<>("id"));
		t1c2.setCellValueFactory(new PropertyValueFactory<>("login"));
		t1c3.setCellValueFactory(new PropertyValueFactory<>("password"));
		t1c4.setCellValueFactory(new PropertyValueFactory<>("firstName"));
		t1c5.setCellValueFactory(new PropertyValueFactory<>("secondName"));
		t1c6.setCellValueFactory(new PropertyValueFactory<>("email"));
		t1c7.setCellValueFactory(new PropertyValueFactory<>("phone"));

		t2c1.setCellValueFactory(new PropertyValueFactory<>("shareId"));
		t2c2.setCellValueFactory(new PropertyValueFactory<>("userId"));
		t2c3.setCellValueFactory(new PropertyValueFactory<>("storageId"));
		t2c4.setCellValueFactory(new PropertyValueFactory<>("accessId"));
	}

	public void b1Add() {
		int id = Integer.valueOf(t1f1.getText());
		String login = t1f2.getText();
		String password = t1f3.getText();
		String firstName = t1f4.getText();
		String secondName = t1f5.getText();
		String email = t1f6.getText();
		String phone = t1f7.getText();
		String query = "insert into USER_INFO (USER_INFO_ID, LOGIN, PASSWORD, FIRST_NAME, SECOND_NAME, EMAIL, PHONE)";

		query += " values (?, ?, ?, ?, ?, ?, ?)";

		UserRow row = new UserRow(id, login, password, firstName, secondName, email, phone);

		t1.getItems().add(row);
		t1f1.clear();
		t1f2.clear();
		t1f3.clear();
		t1f4.clear();
		t1f5.clear();
		t1f6.clear();
		t1f7.clear();

		sqlManager.execute(query, id, login, password, firstName, secondName, email, phone);
	}

	public void b1Del() {
		ObservableList<UserRow> allRows = t1.getItems();
		ObservableList<UserRow> selected = t1.getSelectionModel().getSelectedItems();

		for (UserRow row : selected) {
			allRows.remove(row);

			sqlManager.execute("delete from USER_INFO ui where ui.USER_INFO_ID = ?", row.getId());
		}
	}

	public void b2Update() {
		Integer userId = t2cb1.getSelectionModel().getSelectedItem();
		Integer storageId = t2cb2.getSelectionModel().getSelectedItem();
		Integer accessId = t2cb3.getSelectionModel().getSelectedItem();
		ObservableList<ShareRow> selected = t2.getSelectionModel().getSelectedItems();

		if ((userId == null && storageId == null && accessId == null) || selected.isEmpty()) {
			return;
		}

		for (ShareRow row : selected) {
			if (userId != null) {
				row.setUserId(userId);

				sqlManager.execute("update SHARE_INFO si set si.USER_INFO_ID = ? where si.SHARE_INFO_ID = ?", userId, row.getShareId());
			}

			if (storageId != null) {
				row.setStorageId(storageId);

				sqlManager.execute("update SHARE_INFO si set si.STORAGE_INFO_ID = ? where si.SHARE_INFO_ID = ?", storageId, row.getShareId());
			}

			if (accessId != null) {
				row.setAccessId(accessId);

				sqlManager.execute("update SHARE_INFO si set si.STORAGE_ACCESS_ID = ? where si.SHARE_INFO_ID = ?", accessId, row.getShareId());
			}

			t2.refresh();
			t2cb1.getSelectionModel().clearSelection();
			t2cb2.getSelectionModel().clearSelection();
			t2cb3.getSelectionModel().clearSelection();
		}
	}

	public void b2Clear() {
		t2cb1.getSelectionModel().clearSelection();
		t2cb2.getSelectionModel().clearSelection();
		t2cb3.getSelectionModel().clearSelection();
	}

	public void setSqlManager(SQLManager sqlManager) {
		this.sqlManager = sqlManager;

		t1.setItems(getUserList());
		t2.setItems(getShareList());

		t2cb1.setItems(getUserIdList());
		t2cb2.setItems(getStorageIdList());
		t2cb3.setItems(getAccessIdList());
	}

	public ObservableList<UserRow> getUserList() {
		ObservableList<UserRow> result = FXCollections.observableArrayList();
		List<String> output = new ArrayList<>();

		sqlManager.setOutput(output);
		sqlManager.select("select USER_INFO_ID, LOGIN, PASSWORD, FIRST_NAME, SECOND_NAME, EMAIL, PHONE from USER_INFO order by USER_INFO_ID");

		for (String row : output) {
			String[] parts = row.split("\t");

			result.add(new UserRow(Integer.valueOf(parts[0]), parts[1], parts[2], parts[3], parts[4], parts[5], parts[6]));
		}

		return result;
	}

	public ObservableList<ShareRow> getShareList() {
		ObservableList<ShareRow> result = FXCollections.observableArrayList();
		List<String> output = new ArrayList<>();

		sqlManager.setOutput(output);
		sqlManager.select("select SHARE_INFO_ID, USER_INFO_ID, STORAGE_INFO_ID, STORAGE_ACCESS_ID from SHARE_INFO order by SHARE_INFO_ID");

		for (String row : output) {
			String[] parts = row.split("\t");

			result.add(new ShareRow(Integer.valueOf(parts[0]), Integer.valueOf(parts[1]), Integer.valueOf(parts[2]), Integer.valueOf(parts[3])));
		}

		return result;
	}

	public ObservableList<Integer> getUserIdList() {
		ObservableList<Integer> result = FXCollections.observableArrayList();
		List<String> output = new ArrayList<>();

		sqlManager.setOutput(output);
		sqlManager.select("select USER_INFO_ID from USER_INFO");

		for (String row : output) {
			result.add(Integer.valueOf(row));
		}

		return result;
	}

	public ObservableList<Integer> getStorageIdList() {
		ObservableList<Integer> result = FXCollections.observableArrayList();
		List<String> output = new ArrayList<>();

		sqlManager.setOutput(output);
		sqlManager.select("select STORAGE_INFO_ID from STORAGE_INFO");

		for (String row : output) {
			result.add(Integer.valueOf(row));
		}

		return result;
	}

	public ObservableList<Integer> getAccessIdList() {
		ObservableList<Integer> result = FXCollections.observableArrayList();
		List<String> output = new ArrayList<>();

		sqlManager.setOutput(output);
		sqlManager.select("select STORAGE_ACCESS_ID from STORAGE_ACCESS");

		for (String row : output) {
			result.add(Integer.valueOf(row));
		}

		return result;
	}

	private SQLManager sqlManager;

	@FXML
	private TableView<UserRow> t1;
	@FXML
	private TableColumn<UserRow, Integer> t1c1;
	@FXML
	private TableColumn<UserRow, String> t1c2;
	@FXML
	private TableColumn<UserRow, String> t1c3;
	@FXML
	private TableColumn<UserRow, String> t1c4;
	@FXML
	private TableColumn<UserRow, String> t1c5;
	@FXML
	private TableColumn<UserRow, String> t1c6;
	@FXML
	private TableColumn<UserRow, String> t1c7;
	@FXML
	private TextField t1f1;
	@FXML
	private TextField t1f2;
	@FXML
	private TextField t1f3;
	@FXML
	private TextField t1f4;
	@FXML
	private TextField t1f5;
	@FXML
	private TextField t1f6;
	@FXML
	private TextField t1f7;

	@FXML
	private TableView<ShareRow> t2;
	@FXML
	private TableColumn<UserRow, Integer> t2c1;
	@FXML
	private TableColumn<UserRow, Integer> t2c2;
	@FXML
	private TableColumn<UserRow, Integer> t2c3;
	@FXML
	private TableColumn<UserRow, Integer> t2c4;
	@FXML
	private ComboBox<Integer> t2cb1;
	@FXML
	private ComboBox<Integer> t2cb2;
	@FXML
	private ComboBox<Integer> t2cb3;
}
