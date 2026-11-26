import com.comsol.model.GeomSequence;
import com.comsol.model.Model;
import com.comsol.model.util.ModelUtil;

import java.io.PrintWriter;
import java.util.Locale;

public class MeasureOfficialHeight {
    public static Model run() {
        try {
            String modelPath = System.getProperty("estc.model");
            String outputPath = System.getProperty("estc.output");
            Model model = ModelUtil.load("OfficialESTC", modelPath);
            String componentTag = model.component().tags()[0];
            String geometryTag = model.component(componentTag).geom().tags()[0];
            GeomSequence geometry = model.component(componentTag).geom(geometryTag);

            try (PrintWriter writer = new PrintWriter(outputPath)) {
                writer.println("object,xmin,xmax,ymin,ymax,zmin,zmax,dx,dy,dz");
                for (String objectName : geometry.objectNames()) {
                    double[] box = geometry.object(objectName).getBoundingBox();
                    double dx = box[1] - box[0];
                    double dy = box[3] - box[2];
                    double dz = box[5] - box[4];
                    writer.printf(
                        Locale.US,
                        "%s,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g%n",
                        objectName,
                        box[0], box[1], box[2], box[3], box[4], box[5], dx, dy, dz
                    );
                }
            }
            return model;
        } catch (Exception error) {
            throw new RuntimeException(error);
        }
    }

    public static void main(String[] args) {
        run();
    }
}
