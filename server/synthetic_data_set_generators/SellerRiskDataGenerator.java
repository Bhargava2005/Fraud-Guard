import java.io.FileWriter;
import java.io.IOException;
import java.util.Random;

public class SellerRiskDataGenerator {

    // ✅ FIX 1: Increased from 5000 to 50000
    // 5000 records is too small for CalibratedClassifierCV with cv=5
    // and produces poor probability calibration across all 20 bands
    static final int RECORDS = 50000;

    public static void main(String[] args) {

        String fileName = "../assets/seller_risk_data.csv";
        Random rand = new Random();

        try (FileWriter writer = new FileWriter(fileName)) {

            writer.append(
                "seller_age_days,verification_level,total_orders," +
                "seller_return_rate,seller_dispute_rate,wrong_item_rate," +
                "damaged_item_rate,avg_product_price," +
                "refund_before_inspection,negative_feedback_rate," +
                "seller_risk_label\n"
            );

            for (int i = 0; i < RECORDS; i++) {

                int    sellerAge              = rand.nextInt(2000) + 1;
                int    verificationLevel      = rand.nextInt(3);       // 0=Unverified,1=Partial,2=Full
                int    totalOrders            = rand.nextInt(5000) + 10;
                double returnRate             = rand.nextDouble();
                double disputeRate            = rand.nextDouble() * 0.5;
                double wrongItemRate          = rand.nextDouble() * 0.4;
                double damagedItemRate        = rand.nextDouble() * 0.4;
                double avgProductPrice        = 300 + rand.nextDouble() * 15000;
                int    refundBeforeInspection = rand.nextInt(2);
                double negativeFeedbackRate   = rand.nextDouble() * 0.6;

                // ── Risk Scoring ─────────────────────────────────────────
                double riskScore = 0;
                if (sellerAge < 180)              riskScore += 1.5;
                if (verificationLevel == 0)       riskScore += 2.0;
                if (returnRate > 0.35)            riskScore += 2.0;
                if (disputeRate > 0.20)           riskScore += 1.5;
                if (wrongItemRate > 0.25)         riskScore += 1.2;
                if (damagedItemRate > 0.25)       riskScore += 1.2;
                if (refundBeforeInspection == 1)  riskScore += 1.0;
                if (negativeFeedbackRate > 0.30)  riskScore += 2.0;

                // ── NOISE LAYER ✅ FIX 2 ─────────────────────────────────
                // Seller risk threshold is 5 (lower than other models at 6).
                // This means the borderline zone is narrower and even MORE
                // susceptible to hard boundary overfitting without noise.
                // Gaussian noise ±1.5 creates realistic ambiguity so the
                // calibrated model learns genuine probabilities like 0.48, 0.63
                // instead of extreme 0.00001 / 0.99999 values.
                double noise      = rand.nextGaussian() * 1.5;
                double noisyScore = riskScore + noise;

                // ✅ NOTE: threshold is 5 (more sensitive than other models)
                // This reflects real-world logic: unverified sellers with
                // high return rates are suspicious even with fewer combined flags
                int sellerRiskLabel = noisyScore >= 5 ? 1 : 0;

                writer.append(
                    sellerAge + "," +
                    verificationLevel + "," +
                    totalOrders + "," +
                    String.format("%.2f", returnRate) + "," +
                    String.format("%.2f", disputeRate) + "," +
                    String.format("%.2f", wrongItemRate) + "," +
                    String.format("%.2f", damagedItemRate) + "," +
                    String.format("%.2f", avgProductPrice) + "," +
                    refundBeforeInspection + "," +
                    String.format("%.2f", negativeFeedbackRate) + "," +
                    sellerRiskLabel + "\n"
                );
            }

            System.out.println("✅ " + RECORDS + " seller risk records generated with realistic noise!");

        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
