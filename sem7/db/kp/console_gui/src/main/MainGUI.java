package main;

import java.io.IOException;

import javafx.application.Application;
import javafx.fxml.FXMLLoader;
import javafx.scene.Scene;
import javafx.scene.layout.BorderPane;
import javafx.stage.Stage;

public class MainGUI extends Application {
	@Override
	public void start(Stage stage) throws IOException {
		FXMLLoader loader = new FXMLLoader(getClass().getResource("/resources/main.fxml"));
		BorderPane root = loader.load();
		Scene scene = new Scene(root);
		MainGUIController controller = loader.getController();

		stage.setTitle("Проектирование БД");
		stage.setScene(scene);
		stage.show();
		stage.setResizable(false);

		sqlManager = new SQLManager();
		sqlManager.connect();

		controller.setSqlManager(sqlManager);
	}

	@Override
	public void stop() {
		sqlManager.disconnect();
	}

	public static void main(String[] args) {
		launch(args);
	}

	private SQLManager sqlManager;
}
