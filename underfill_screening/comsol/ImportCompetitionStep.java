import com.comsol.model.Model;
import com.comsol.model.util.ModelUtil;

public class ImportCompetitionStep {
  public static Model run() throws java.io.IOException {
    String inputFile = "C:\\COMSOL_IMPORT\\estc_model_solder.stp";
    String outputFile = "C:\\COMSOL_IMPORT\\estc_model_solder_original_imported.mph";

    Model model = ModelUtil.create("Model");
    model.label("estc_model_solder_original_imported.mph");
    model.modelPath("C:\\COMSOL_IMPORT");
    model.component().create("comp1", true);
    model.component("comp1").geom().create("geom1", 3);
    model.component("comp1").geom("geom1").feature().create("imp1", "Import");
    model.component("comp1").geom("geom1").feature("imp1").set("filename", inputFile);
    model.component("comp1").geom("geom1").feature("imp1").importData();
    model.component("comp1").geom("geom1").run();
    model.save(outputFile);
    return model;
  }

  public static void main(String[] args) throws java.io.IOException {
    run();
  }
}
