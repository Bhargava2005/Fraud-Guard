import java.io.FileWriter;
import java.io.IOException;
import java.util.Random;

public class ProductRiskDataGenerator {

    // ✅ FIX 1: Increased to 50000. Original said RECORDS=15000
    // but printed "5000 records generated" — misleading and too small
    // for CalibratedClassifierCV with cv=5
    static final int RECORDS = 50000;

    public static void main(String[] args) {

        String fileName = "../assets/product_risk_data.csv";
        Random rand = new Random();

        try (FileWriter writer = new FileWriter(fileName)) {

            writer.append(
                "product_price,product_category,return_rate," +
                "fraud_return_rate,avg_days_to_return," +
                "serial_tracked,counterfeit_risk,fragile," +
                "seller_product_risk,discount_percentage," +
                "product_risk_label\n"
            );

            for (int i = 0; i < RECORDS; i++) {

                double productPrice      = 200 + rand.nextDouble() * 30000;
                int    productCategory   = rand.nextInt(3);        // 0=Low, 1=Medium, 2=High
                double returnRate        = rand.nextDouble();
                double fraudReturnRate   = rand.nextDouble() * 0.5;
                int    avgDaysToReturn   = rand.nextInt(30) + 1;
                int    serialTracked     = rand.nextInt(2);
                double counterfeitRisk   = rand.nextDouble();
                int    fragile           = rand.nextInt(2);
                double sellerProductRisk = rand.nextDouble();
                double discountPct       = rand.nextDouble() * 0.8;

                // ── Risk Scoring ─────────────────────────────────────────
                double riskScore = 0;
                if (productPrice > 15000)      riskScore += 2.0;
                if (productCategory == 2)      riskScore += 1.5;
                if (returnRate > 0.35)         riskScore += 2.0;
                if (fraudReturnRate > 0.20)    riskScore += 2.0;
                if (avgDaysToReturn > 20)      riskScore += 1.0;
                if (serialTracked == 0)        riskScore += 1.5;
                if (counterfeitRisk > 0.5)     riskScore += 1.5;
                if (fragile == 1)              riskScore += 1.0;
                if (sellerProductRisk > 0.6)   riskScore += 1.5;
                if (discountPct > 0.5)         riskScore += 1.0;

                // ── NOISE LAYER ✅ FIX 2 ─────────────────────────────────
                // Without noise: every record with score ≥ 6 gets label=1
                // and score < 6 gets label=0 with ZERO overlap between classes.
                // The model memorises these hard cutoffs and outputs extreme
                // probabilities (0.000001 or 0.999999).
                // Gaussian noise ±1.5 flips borderline records randomly,
                // creating realistic overlap so the calibrated model learns
                // genuine probability values like 0.43, 0.61, 0.78 etc.
                double noise      = rand.nextGaussian() * 1.5;
                double noisyScore = riskScore + noise;

                int productRiskLabel = noisyScore >= 6 ? 1 : 0;

                writer.append(
                    String.format("%.2f", productPrice) + "," +
                    productCategory + "," +
                    String.format("%.2f", returnRate) + "," +
                    String.format("%.2f", fraudReturnRate) + "," +
                    avgDaysToReturn + "," +
                    serialTracked + "," +
                    String.format("%.2f", counterfeitRisk) + "," +
                    fragile + "," +
                    String.format("%.2f", sellerProductRisk) + "," +
                    String.format("%.2f", discountPct) + "," +
                    productRiskLabel + "\n"
                );
            }

            System.out.println("✅ " + RECORDS + " product risk records generated with realistic noise!");

        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
