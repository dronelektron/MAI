package application;

import javafx.fxml.FXMLLoader;
import javafx.application.Application;
import javafx.scene.Scene;
import javafx.scene.layout.BorderPane;
import javafx.stage.Stage;

public class Main extends Application {
	@Override
	public void start(Stage stage) throws Exception {
		FXMLLoader loader = new FXMLLoader(getClass().getResource("../resources/main_window.fxml"));
		BorderPane root = loader.load();
		Scene scene = new Scene(root);

		stage.setTitle("Компьютерная графика - лабораторная работа 3");
		stage.setScene(scene);
		stage.show();
		stage.setMinWidth(scene.getWindow().getWidth());
		stage.setMinHeight(scene.getWindow().getHeight());
	}

	public static void main(String[] args) {
		launch(args);
	}
}
