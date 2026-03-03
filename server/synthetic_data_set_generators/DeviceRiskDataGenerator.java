import java.io.FileWriter;
import java.io.IOException;
import java.util.Random;

public class DeviceRiskDataGenerator {

    static final int RECORDS = 50000; // Increased from 5000 for better model training

    public static void main(String[] args) {

        String fileName = "../assets/device_risk_data.csv";
        Random rand = new Random();

        try (FileWriter writer = new FileWriter(fileName)) {

            writer.append(
                "device_age_days,device_type,accounts_per_device," +
                "emulator_detected,rooted_or_jailbroken,vpn_used," +
                "ip_changes_24h,geo_distance_km,login_frequency," +
                "failed_login_attempts,device_risk_label\n"
            );

            for (int i = 0; i < RECORDS; i++) {

                int deviceAge         = rand.nextInt(1500) + 1;
                int deviceType        = rand.nextInt(3);       // 0=Mobile,1=Desktop,2=Tablet
                int accountsPerDevice = rand.nextInt(6);
                int emulatorDetected  = rand.nextDouble() < 0.08 ? 1 : 0;
                int rootedDevice      = rand.nextDouble() < 0.10 ? 1 : 0;
                int vpnUsed           = rand.nextDouble() < 0.25 ? 1 : 0;
                int ipChanges24h      = rand.nextInt(10);
                int geoDistanceKm     = rand.nextInt(3000);
                int loginFrequency    = rand.nextInt(50) + 1;
                int failedLogins      = rand.nextInt(8);

                // ── Risk Scoring ─────────────────────────────────────────
                double riskScore = 0;
                if (deviceAge < 60)            riskScore += 1.5;
                if (accountsPerDevice >= 3)    riskScore += 2.0;
                if (emulatorDetected == 1)     riskScore += 2.5;
                if (rootedDevice == 1)         riskScore += 2.0;
                if (vpnUsed == 1)              riskScore += 1.5;
                if (ipChanges24h >= 5)         riskScore += 1.5;
                if (geoDistanceKm > 500)       riskScore += 1.2;
                if (loginFrequency > 30)       riskScore += 1.0;
                if (failedLogins >= 4)         riskScore += 1.5;

                // ── NOISE LAYER (prevents overconfident 0.00001/0.99999 predictions) ──
                // Gaussian noise ±1.5 creates realistic borderline overlap
                // so model outputs calibrated probabilities instead of hard boundaries
                double noise      = rand.nextGaussian() * 1.5;
                double noisyScore = riskScore + noise;

                int deviceRiskLabel = noisyScore >= 6 ? 1 : 0;

                writer.append(
                    deviceAge + "," +
                    deviceType + "," +
                    accountsPerDevice + "," +
                    emulatorDetected + "," +
                    rootedDevice + "," +
                    vpnUsed + "," +
                    ipChanges24h + "," +
                    geoDistanceKm + "," +
                    loginFrequency + "," +
                    failedLogins + "," +
                    deviceRiskLabel + "\n"
                );
            }

            System.out.println("✅ " + RECORDS + " device risk records generated with realistic noise!");

        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
