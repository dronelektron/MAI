package main;

import javafx.application.Application;
import javafx.fxml.FXMLLoader;
import javafx.scene.Scene;
import javafx.scene.layout.BorderPane;
import javafx.stage.Stage;
import java.io.IOException;

public class Main extends Application {
	@Override
	public void start(Stage stage) throws IOException {
		FXMLLoader loader = new FXMLLoader(getClass().getResource("/resources/main.fxml"));
		BorderPane root = loader.load();
		Scene scene = new Scene(root);

		scene.getStylesheets().add(getClass().getResource("/resources/styles/chart.css").toExternalForm());

		stage.setTitle("Численные методы - лабораторная работа 5");
		stage.setScene(scene);
		stage.show();
		stage.setMinWidth(scene.getWindow().getWidth());
		stage.setMinHeight(scene.getWindow().getHeight());
	}

	public static void main(String[] args) {
		launch(args);
	}
}
