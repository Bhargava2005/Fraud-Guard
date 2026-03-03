import java.io.FileWriter;
import java.io.IOException;
import java.util.Random;

public class LogisticsRiskDataGenerator {

    static final int RECORDS = 50000; // Increased from 5000 for better calibration

    public static void main(String[] args) {

        String fileName = "../assets/logistics_risk_data.csv";
        Random rand = new Random();

        try (FileWriter writer = new FileWriter(fileName)) {

            writer.append(
                "courier_risk_score,delivery_attempts,delivery_delay_days," +
                "otp_confirmation,delivery_photo,pickup_delay_days," +
                "pickup_attempts,tamper_route,distance_km," +
                "weight_mismatch,logistics_risk_label\n"
            );

            for (int i = 0; i < RECORDS; i++) {

                double courierRisk       = rand.nextDouble();
                int deliveryAttempts     = rand.nextInt(5) + 1;
                int deliveryDelay        = rand.nextInt(15);
                int otpConfirmed         = rand.nextDouble() < 0.75 ? 1 : 0;
                int deliveryPhoto        = rand.nextDouble() < 0.70 ? 1 : 0;
                int pickupDelay          = rand.nextInt(15);
                int pickupAttempts       = rand.nextInt(4) + 1;
                int tamperRoute          = rand.nextDouble() < 0.20 ? 1 : 0;
                int distanceKm           = rand.nextInt(2500);
                int weightMismatch       = rand.nextDouble() < 0.15 ? 1 : 0;

                // ── Risk Scoring ─────────────────────────────────────────
                double riskScore = 0;
                if (courierRisk > 0.6)       riskScore += 2.0;
                if (deliveryAttempts >= 3)   riskScore += 1.5;
                if (deliveryDelay > 5)       riskScore += 1.2;
                if (otpConfirmed == 0)       riskScore += 1.5;
                if (deliveryPhoto == 0)      riskScore += 1.2;
                if (pickupDelay > 5)         riskScore += 1.0;
                if (pickupAttempts >= 3)     riskScore += 1.2;
                if (tamperRoute == 1)        riskScore += 1.5;
                if (distanceKm > 800)        riskScore += 1.0;
                if (weightMismatch == 1)     riskScore += 2.0;

                // ── NOISE LAYER ───────────────────────────────────────────
                // Gaussian noise ±1.5 creates realistic borderline overlap
                // so model outputs calibrated probabilities (e.g. 0.63)
                // instead of extreme values (0.00001 or 0.99999)
                double noise      = rand.nextGaussian() * 1.5;
                double noisyScore = riskScore + noise;

                int logisticsRiskLabel = noisyScore >= 6 ? 1 : 0;

                writer.append(
                    String.format("%.2f", courierRisk) + "," +
                    deliveryAttempts + "," +
                    deliveryDelay + "," +
                    otpConfirmed + "," +
                    deliveryPhoto + "," +
                    pickupDelay + "," +
                    pickupAttempts + "," +
                    tamperRoute + "," +
                    distanceKm + "," +
                    weightMismatch + "," +
                    logisticsRiskLabel + "\n"
                );
            }

            System.out.println("✅ " + RECORDS + " logistics risk records generated with realistic noise!");

        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
