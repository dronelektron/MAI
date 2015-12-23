import javax.xml.transform.*;
import javax.xml.transform.stream.StreamResult;
import javax.xml.transform.stream.StreamSource;
import java.io.File;
import java.io.IOException;
import java.net.URISyntaxException;

public class Comp {
	public static void main(String[] args) {
		if (args.length < 3) {
			System.out.println("Usage: java Comp file.xml file.xsd file.html");

			return;
		}

		try {
			TransformerFactory factory = TransformerFactory.newInstance();
			Source text = new StreamSource(new File(args[0]));
			Source xslt = new StreamSource(new File(args[1]));
			Transformer transformer = factory.newTransformer(xslt);

			transformer.transform(text, new StreamResult(new File(args[2])));

			System.out.println("Transformation success");
		} catch (TransformerException e) {
			System.out.println("Transformation error");
			System.out.println(e.getMessage());
		}
	}
}