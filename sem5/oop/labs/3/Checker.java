import java.io.*;
import javax.xml.transform.Source;
import javax.xml.transform.stream.StreamSource;
import javax.xml.validation.*;
import org.xml.sax.SAXException;

public class Checker {
	public static void main(String[] args) throws SAXException, IOException {
		if (args.length < 2) {
			System.out.println("Usage: java Checker file.xml file.xsd");

			return;
		}
		
		SchemaFactory factory = SchemaFactory.newInstance("http://www.w3.org/2001/XMLSchema");
		File schemaLocation = new File(args[1]);
		Schema schema = factory.newSchema(schemaLocation);
		Validator validator = schema.newValidator();
		Source source = new StreamSource(args[0]);
		
		try {
			validator.validate(source);

			System.out.println(args[0] + " is valid.");
		} catch (SAXException e) {
			System.out.println(args[0] + " is not valid because ");
			System.out.println(e.getMessage());
		}
	}
}
