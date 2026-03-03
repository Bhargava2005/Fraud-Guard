import java.io.FileWriter;
import java.io.IOException;
import java.util.Random;

public class OrderRiskDataGenerator {

    static final int RECORDS = 50000; // ✅ FIX 1: increased from 5000 for calibration

    public static void main(String[] args) {

        String fileName = "../assets/order_risk_data.csv";
        Random rand = new Random();

        try (FileWriter writer = new FileWriter(fileName)) {

            writer.append(
                "order_value,item_quantity,identical_items," +
                "payment_method,failed_payment_attempts," +
                "order_hour,checkout_time_sec,address_mismatch," +
                "order_velocity,customer_tenure_days," +
                "order_risk_label\n"
            );

            for (int i = 0; i < RECORDS; i++) {

                double orderValue    = 300 + rand.nextDouble() * 40000;
                int itemQuantity     = rand.nextInt(10) + 1;
                int identicalItems   = rand.nextInt(itemQuantity + 1);
                int paymentMethod    = rand.nextInt(2);   // 0=Prepaid, 1=COD
                int failedPayments   = rand.nextInt(5);
                int orderHour        = rand.nextInt(24);  // 0–23
                int checkoutTime     = rand.nextInt(600) + 10;
                int addressMismatch  = rand.nextInt(2);
                int orderVelocity    = rand.nextInt(8) + 1;
                int customerTenure   = rand.nextInt(1500) + 1;

                // ── Risk Scoring ─────────────────────────────────────────
                double riskScore = 0;
                if (orderValue > 20000)     riskScore += 2.0;
                if (itemQuantity >= 5)      riskScore += 1.5;
                if (identicalItems >= 3)    riskScore += 1.5;
                if (paymentMethod == 1)     riskScore += 1.2;
                if (failedPayments >= 3)    riskScore += 1.5;
                if (orderHour < 5)          riskScore += 1.0;  // ✅ FIX 2: removed
                                                                // orderHour > 23 (impossible
                                                                // since nextInt(24) = 0–23)
                if (checkoutTime < 30)      riskScore += 1.2;
                if (addressMismatch == 1)   riskScore += 1.5;
                if (orderVelocity >= 5)     riskScore += 1.5;
                if (customerTenure < 60)    riskScore += 1.5;

                // ── NOISE LAYER ✅ FIX 3 ─────────────────────────────────
                // Without noise every score ≥ 6 → label=1 and score < 6 → label=0
                // with zero overlap. Model memorizes hard boundaries and outputs
                // probabilities of 0.000001 or 0.999999 instead of realistic values.
                // Gaussian noise ±1.5 creates a natural borderline zone so the model
                // learns true probabilities like 0.52, 0.67, 0.38 etc.
                double noise      = rand.nextGaussian() * 1.5;
                double noisyScore = riskScore + noise;

                int orderRiskLabel = noisyScore >= 6 ? 1 : 0;

                writer.append(
                    String.format("%.2f", orderValue) + "," +
                    itemQuantity + "," +
                    identicalItems + "," +
                    paymentMethod + "," +
                    failedPayments + "," +
                    orderHour + "," +
                    checkoutTime + "," +
                    addressMismatch + "," +
                    orderVelocity + "," +
                    customerTenure + "," +
                    orderRiskLabel + "\n"
                );
            }

            System.out.println("✅ " + RECORDS + " order risk records generated with realistic noise!");

        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
