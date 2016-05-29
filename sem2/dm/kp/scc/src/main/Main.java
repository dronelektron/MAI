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
		FXMLLoader loader = new FXMLLoader(getClass().getResource("resources/main_window.fxml"));
		BorderPane root = loader.load();
		Scene scene = new Scene(root);

		stage.setTitle("Дискретная математика - разложение графа на максимально связанные подграфы");
		stage.setScene(scene);
		stage.show();
		stage.setMinWidth(scene.getWindow().getWidth());
		stage.setMinHeight(scene.getWindow().getHeight());
	}

	public static void main(String[] args) {
		launch(args);
	}
}
